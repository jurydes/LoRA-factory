from src.schemas import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model

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

# preprocessing and loading dataset
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

    return tokenized

# training LoRA and logging the results
def train_and_log(model, dataset, config: TrainConfig):
    pass

# checking if model is ok
def val_check(model, dataset, config: TrainConfig):
    pass

# main function
def run_lora_pipeline(config: TrainConfig):
    tokenizer, model = load_model(config)
    dataset = create_dataset(config, tokenizer)

    # blank for now 
    trained = train_and_log(model, dataset, config)
    val_check(trained, dataset, config)

    return f"Модель {config.model_name} обучена и загружена в {config.output_dir}"