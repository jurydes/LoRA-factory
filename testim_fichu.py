import os
import sys
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from settings import TrainConfig
from load_model import load_model
from inference import compute_metrics

EXPANDED_PROMPT = (
    "Перефразируй следующий токсичный или оскорбительный текст в нейтральный, сохранив исходный смысл. "
    "Убери ненормативную лексику, агрессию и оскорбления, но сохрани суть высказывания."
)

LORA_PATH = "./qwen3_lora"

config = TrainConfig()
config.model_name = "google/gemma-2b"
config.num_eval_samples = 999_999  # полная тестовая выборка

ds = load_dataset("s-nlp/ru_paradetox")
split = ds["train"].train_test_split(test_size=config.val_split, seed=42)
test_ds = split["test"].rename_column("ru_toxic_comment", "input").rename_column("ru_neutral_comment", "target")
test_ds = test_ds.select_columns(["input", "target"])

# Прогоны 1 и 2 — base model с новым (необученным) LoRA-адаптером
tokenizer, base_model = load_model(config)

print("\n>>> [1/3] Без промпта (base model)...")
m_base = compute_metrics(tokenizer, base_model, test_ds, config, label="без промпта")

print("\n>>> [2/3] С расширенным промптом (base model)...")
m_prompt = compute_metrics(tokenizer, base_model, test_ds, config, label="с промптом", prompt_prefix=EXPANDED_PROMPT)

# Прогон 3 — чистая base-модель + сохранённые веса LoRA
print(f"\n>>> [3/3] Загружаем чистую base-модель + веса LoRA из {LORA_PATH}...")
del base_model
torch.cuda.empty_cache()

raw_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16 if config.bf16 else "auto",
    device_map={"": config.device},
)
trained_model = PeftModel.from_pretrained(raw_model, LORA_PATH)
trained_model.eval()

print(">>> Прогон с обученными весами БЕЗ промпта...")
m_trained = compute_metrics(tokenizer, trained_model, test_ds, config, label="LoRA без промпта")

print("\n=== Итоговое сравнение ===")
print(f"{'Метрика':<15} {'Base (без промпта)':>20} {'Base (с промптом)':>20} {'LoRA (без промпта)':>20}")
print("-" * 77)
for key, fmt in [("exact_match", ".4f"), ("bleu", ".2f"), ("chrf", ".4f"), ("bert_score_f1", ".4f")]:
    print(f"{key:<15} {m_base[key]:>20{fmt}} {m_prompt[key]:>20{fmt}} {m_trained[key]:>20{fmt}}")
