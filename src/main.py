import logging
import multiprocessing
import traceback
import asyncio
from settings import TrainConfig, HYPERPARAMETER_PRESETS
from load_model import load_model
from dataset_work import create_dataset, check_available_datasets, detect_columns
from train import train_and_log
from inference import val_check, compute_metrics
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.getLogger("httpx").setLevel(logging.WARNING)


def choose_hyperparameters(config: TrainConfig) -> TrainConfig:
    """Интерактивный выбор гиперпараметров из пресетов или ручной ввод. Возвращает обновлённый config."""
    print("\nВыберите конфигурацию гиперпараметров:")
    for key, preset in HYPERPARAMETER_PRESETS.items():
        print(f"  [{key}] {preset['name']}: {preset['description']}")

    choice = input("\nВаш выбор [1-4]: ").strip()
    preset = HYPERPARAMETER_PRESETS.get(choice, HYPERPARAMETER_PRESETS["2"])

    if choice not in ['1', '2', '3', '4']:
        print("Выбран некорректный конфиг, выставляются параметры по умолчанию")
    elif choice == "4":
        def ask(name, current):
            val = input(f"  {name} [{current}]: ").strip()
            return val if val else str(current)

        config.num_train_epochs = int(ask("num_train_epochs", config.num_train_epochs))
        config.learning_rate = float(ask("learning_rate", config.learning_rate))
        config.per_device_train_batch_size = int(ask("batch_size", config.per_device_train_batch_size))
        config.lora_r = int(ask("lora_r", config.lora_r))
        config.lora_alpha = int(ask("lora_alpha", config.lora_alpha))
    else:
        for k, v in preset["params"].items():
            setattr(config, k, v)
        print(f"\nВыбрана конфигурация: {preset['name']}")

    print("\nИтоговые параметры:")
    print(f"  epochs={config.num_train_epochs}, lr={config.learning_rate}, "
          f"batch={config.per_device_train_batch_size}, "
          f"lora_r={config.lora_r}, lora_alpha={config.lora_alpha}")
    return config


def _training_worker(config: TrainConfig, datasets_info: list[tuple[str, str, str]]):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    from config_gen import suggest_config
    try:
        tokenizer, model = load_model(config)
        raw, dataset = create_dataset(config, tokenizer, datasets_info)

        metrics_before = compute_metrics(tokenizer, model, raw["test"], config, label="ДО обучения")

        suggested, reasons = suggest_config(config, len(raw["train"]), metrics_before["bert_score_f1"])

        print("\nПредложенная конфигурация на основе эвристик:")
        for reason in reasons:
            print(f"  • {reason}")
        print(f"\n  epochs={suggested.num_train_epochs}, lr={suggested.learning_rate}", end="")
        if config.adapter_type == "lora":
            print(f", lora_r={suggested.lora_r}, lora_alpha={suggested.lora_alpha}")
        else:
            print()
        print("\n[1] Использовать предложенную конфигурацию (по умолчанию)")
        print("[2] Ввести параметры вручную")
        if input("Ваш выбор [1-2]: ").strip() == "2":
            config = choose_hyperparameters(config)
        else:
            config = suggested
            print("Используется предложенная конфигурация.")

        trained = train_and_log(tokenizer, model, dataset, config)
        metrics_after = compute_metrics(tokenizer, trained, raw["test"], config, label="ПОСЛЕ обучения")

        logger.info(
            "Сравнение: ExactMatch %.4f → %.4f, BERTScore F1 %.4f → %.4f",
            metrics_before["exact_match"], metrics_after["exact_match"],
            metrics_before["bert_score_f1"], metrics_after["bert_score_f1"],
        )

        val_check(tokenizer, trained, raw["test"], config)

    except Exception as e:
        logger.error("Критическая ошибка в процессе обучения: %s", e)
        traceback.print_exc()
        raise SystemExit(1)


def run_lora_pipeline(config: TrainConfig, datasets_info: list[tuple[str, str, str]]) -> str:
    """Запускает полный пайплайн обучения в отдельном процессе (spawn) для безопасной работы с CUDA.

    Возвращает строку с результатом или бросает RuntimeError при аварийном завершении.
    """
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(target=_training_worker, args=(config, datasets_info))
    process.start()
    process.join()

    if process.exitcode == 0:
        return f"Модель {config.model_name} успешно обучена, проверена и сохранена в {config.output_dir}"
    else:
        raise RuntimeError(f"Процесс обучения завершился аварийно с кодом: {process.exitcode}")


async def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    config = TrainConfig()

    MODEL_PRESETS = {
        "1": ("google/gemma-2b",    "Google Gemma 2B (по умолчанию)"),
        "2": ("ai-forever/mGPT",    "mGPT 1.3B — обучена на русском"),
        "3": ("Qwen/Qwen3.5-2B",   "Qwen3.5 2B — может отказываться от токсичных текстов"),
    }
    print("\nВыберите модель:")
    for key, (model_id, desc) in MODEL_PRESETS.items():
        print(f"  [{key}] {model_id} — {desc}")
    print(f"  [Enter] по умолчанию: {MODEL_PRESETS['1'][0]}")
    print("  [другое] введите свой HuggingFace id")

    choice = input("\nВаш выбор: ").strip()
    if choice in MODEL_PRESETS:
        config.model_name = MODEL_PRESETS[choice][0]
    elif choice:
        config.model_name = choice
    else:
        config.model_name = MODEL_PRESETS["1"][0]
    print(f"Выбрана модель: {config.model_name}")

    print("\nВыберите метод адаптации:")
    print("  [1] LoRA — стандартный, хорошее качество (по умолчанию)")
    print("  [2] P-tuning — только виртуальные токены, минимум параметров")
    adapter_choice = input("\nВаш выбор [1-2]: ").strip()
    if adapter_choice == "2":
        config.adapter_type = "p-tuning"
        print("Выбран P-tuning")
    else:
        config.adapter_type = "lora"
        print("Выбрана LoRA")

    task_description = input("Опишите задачу, которую вы решаете: ").strip()

    available_datasets = await check_available_datasets(task_description, config)
    if available_datasets:
        print("\nВведите id датасетов для обучения через запятую (можно несколько):")
        chosen_raw = input("dataset id(s): ").strip()
        dataset_ids = [d.strip() for d in chosen_raw.split(",") if d.strip()]
    else:
        dataset_ids = [config.dataset_name]

    datasets_info: list[tuple[str, str, str]] = []
    for dataset_id in dataset_ids:
        print(f"\nОпределяем колонки датасета '{dataset_id}'...")
        while True:
            try:
                ds_peek = hf_load_dataset(dataset_id, split="train[:1]")
                input_col, target_col = await detect_columns(dataset_id, ds_peek.column_names)

                if not input_col or not target_col:
                    raise ValueError(f"Не удалось определить колонки. Колонки датасета: {ds_peek.column_names}")

                print(f"  input: {input_col}, target: {target_col}")
                datasets_info.append((dataset_id, input_col, target_col))
                break

            except (RuntimeError, ValueError) as e:
                print(f"\n{e}")
                dataset_id = input("Введите id другого датасета: ").strip()

    model_slug = config.model_name.split("/")[-1]
    dataset_slug = "+".join(ds_id.split("/")[-1] for ds_id, _, _ in datasets_info)
    config.output_dir = f"./{model_slug}_{dataset_slug}"

    print(run_lora_pipeline(config, datasets_info))


if __name__ == '__main__':
    asyncio.run(main())
