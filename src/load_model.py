from src.schemas import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# loading model
def load_model(config: TrainConfig):
    """
    Loads model and tokenizer from pretrained
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type
    )

    lora = get_peft_model(model, lora_config)

    return tokenizer, lora