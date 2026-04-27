import copy
from settings import TrainConfig


def suggest_config(config: TrainConfig, dataset_size: int, baseline_bert: float) -> tuple[TrainConfig, list[str]]:
    """Предлагает гиперпараметры на основе размера датасета и базовых метрик модели.

    Возвращает (новый TrainConfig с предложенными параметрами, список строк с обоснованием).
    """
    suggested = copy.copy(config)
    reasons = []

    if dataset_size < 500:
        suggested.learning_rate = 5e-5
        suggested.num_train_epochs = 12
        if config.adapter_type == "lora":
            suggested.lora_r = 4
            suggested.lora_alpha = 8
        reasons.append(f"датасет малый ({dataset_size} примеров) → lr=5e-5, эпох=12, r=4 (риск переобучения)")
    elif dataset_size < 5000:
        suggested.learning_rate = 1e-4
        suggested.num_train_epochs = 6
        if config.adapter_type == "lora":
            suggested.lora_r = 8
            suggested.lora_alpha = 16
        reasons.append(f"датасет средний ({dataset_size} примеров) → lr=1e-4, эпох=6, r=8")
    else:
        suggested.learning_rate = 5e-5
        suggested.num_train_epochs = 8
        if config.adapter_type == "lora":
            suggested.lora_r = 16
            suggested.lora_alpha = 32
        reasons.append(f"датасет большой ({dataset_size} примеров) → lr=5e-5, эпох=8, r=16")

    if baseline_bert > 0.75:
        suggested.learning_rate /= 2
        suggested.num_train_epochs = max(suggested.num_train_epochs - 3, 1)
        reasons.append(f"модель уже справляется (BERTScore={baseline_bert:.2f}) → lr÷2, эпох-3 (лёгкое дообучение)")
    elif baseline_bert < 0.5:
        suggested.num_train_epochs += 4
        if config.adapter_type == "lora":
            suggested.lora_r = min(suggested.lora_r * 2, 32)
            suggested.lora_alpha = suggested.lora_r * 2
        reasons.append(f"модель плохо справляется (BERTScore={baseline_bert:.2f}) → эпох+4, r×2 (серьёзная адаптация)")
    else:
        reasons.append(f"базовые метрики в норме (BERTScore={baseline_bert:.2f}) → без корректировок")

    return suggested, reasons
