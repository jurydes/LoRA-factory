import torch
import asyncio
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset as hf_load_dataset
from dataset_work import create_dataset, detect_columns
from settings import TrainConfig
from bert_score import score as bert_score_fn
import sacrebleu

config = TrainConfig()

use_adapter = input("Загрузить обученный адаптер? [y/n] (Enter = y): ").strip().lower() != "n"

tokenizer = AutoTokenizer.from_pretrained(config.output_dir if use_adapter else config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16)
if use_adapter:
    model = PeftModel.from_pretrained(model, config.output_dir)
    print("Загружена обученная модель с адаптером")
else:
    print("Загружена базовая модель без адаптера")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()


def predict(inp: str, max_new_tokens=48) -> str:
    prompt = f"Input: {inp}\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_full(test_split):
    predictions, references = [], []
    for example in tqdm(test_split, desc="Evaluating"):
        predictions.append(predict(example["input"]))
        references.append(example["target"])

    rouge_l = sacrebleu.corpus_chrf(predictions, [references]).score / 100

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    _, _, f1 = bert_score_fn(predictions, references, model_type="bert-base-multilingual-cased", verbose=False)

    print(f"\nРезультаты на {len(predictions)} примерах:")
    print(f"  BLEU:          {bleu:.2f}")
    print(f"  chrF:          {rouge_l:.4f}")
    print(f"  BERTScore F1:  {f1.mean().item():.4f}")
    return predictions, references


async def main():
    cfg = TrainConfig()

    dataset_name = input("Dataset id (Enter = config.dataset_name): ").strip() or cfg.dataset_name

    ds_peek = hf_load_dataset(dataset_name, split="train[:1]")
    input_col, target_col = await detect_columns(dataset_name, ds_peek.column_names)
    print(f"input: {input_col}, target: {target_col}")

    raw, _ = create_dataset(cfg, tokenizer, [(dataset_name, input_col, target_col)])
    test_split = raw["test"]

    mode = input("Режим: [1] примеры, [2] полная оценка (Enter = 1): ").strip()

    if mode == "2":
        evaluate_full(test_split)
    else:
        n = int(input("Сколько примеров прогнать? ").strip() or "5")
        for i in range(min(n, len(test_split))):
            inp = test_split[i]["input"]
            target = test_split[i]["target"]
            prediction = predict(inp)

            print(f"\n[{i+1}]")
            print(f"Input:      {inp}")
            print(f"Target:     {target}")
            print(f"Prediction: {prediction}")


if __name__ == "__main__":
    asyncio.run(main())
