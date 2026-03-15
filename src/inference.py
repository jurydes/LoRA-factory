from src.schemas import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# checking if model is ok
def val_check(tokenizer, model, dataset, config: TrainConfig):
    model.eval()

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

        print(f"[{i+1}]")
        print(f"Input:      {inp}")
        print(f"Target:     {target}")
        print(f"Prediction: {prediction}")
        print()  