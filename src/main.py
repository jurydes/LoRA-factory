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

    return split, tokenized

# training LoRA and logging the results
def train_and_log(tokenizer, model, dataset, config: TrainConfig):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
        hub_token=config.hub_token if config.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return model

# checking if model is ok
def val_check(tokenizer, model, dataset, config: TrainConfig):
    model.eval()
    lines = []

    for i in range(config.num_val_samples):
        inp = dataset[i]["input"]
        target = dataset[i]["target"]

        prompt = f"Input: {inp}\nTarget:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
            )

        prediction = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        lines.append(f"[{i+1}]\nInput:      {inp}\nTarget:     {target}\nPrediction: {prediction}\n\n")

    with open(config.log_file, "w") as f:
        f.writelines(lines)

# main function
def run_lora_pipeline(config: TrainConfig):
    tokenizer, model = load_model(config)
    raw, dataset = create_dataset(config, tokenizer)

    trained = train_and_log(tokenizer, model, dataset, config)
    
    val_check(tokenizer, trained, raw["test"], config)

    return f"Модель {config.model_name} обучена и загружена в {config.output_dir}"