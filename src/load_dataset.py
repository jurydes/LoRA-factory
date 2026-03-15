from src.schemas import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

def make_preprocess(tokenizer, max_length):
    def preprocess(batch):
        texts = [
            f"Input: {inp}\nTarget: {tgt}"
            for inp, tgt in zip(batch["input"], batch["target"])
        ]
        tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return preprocess

def create_dataset(config: TrainConfig, tokenizer):
    dataset = load_dataset("zamal/github-meta-data")
    split = dataset["train"].train_test_split(test_size=config.val_split, seed=42)

    tokenized = split.map(make_preprocess(tokenizer, config.max_length), batched=True, remove_columns=["input", "target"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return split, tokenized