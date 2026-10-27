import logging
from settings import TrainConfig
import torch
import sacrebleu
from bert_score import score as bert_score_fn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def val_check(tokenizer, model, dataset, config: TrainConfig, input_col="input", target_col="target") -> None:
    """Печатает config.num_val_samples примеров с предсказаниями модели для ручной проверки."""
    model.eval()

    for i in range(config.num_val_samples):
        inp = dataset[i][input_col]
        target = dataset[i][target_col]

        prompt = f"Input: {inp}\nTarget:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.max_new_tokens,
                repetition_penalty=1.3,
            )

        prediction = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

        print(f"[{i+1}]")
        print(f"Input:      {inp}")
        print(f"Target:     {target}")
        print(f"Prediction: {prediction}\n")


def _generate_prediction(tokenizer, model, inp, config, prompt_prefix=""):
    prefix = f"{prompt_prefix}\n\n" if prompt_prefix else ""
    prompt = f"{prefix}Input: {inp}\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.split("\n")[0].strip()


def compute_metrics(tokenizer, model, dataset, config: TrainConfig, input_col="input", target_col="target", label="", prompt_prefix="") -> dict:
    """Считает ExactMatch и BERTScore F1 на config.num_eval_samples примерах. Возвращает dict с метриками."""
    model.eval()
    n = min(len(dataset), config.num_eval_samples)

    predictions = []
    references = []
    for i in tqdm(range(n), desc=label or "eval"):
        pred = _generate_prediction(tokenizer, model, dataset[i][input_col], config, prompt_prefix)
        ref = str(dataset[i][target_col]).strip()
        predictions.append(pred)
        references.append(ref)

    exact_match = sum(p == r for p, r in zip(predictions, references)) / n

    safe_preds = [p if p.strip() else "нет" for p in predictions]
    bleu = sacrebleu.corpus_bleu(safe_preds, [references]).score
    chrf = sacrebleu.corpus_chrf(safe_preds, [references]).score / 100

    _, _, f1 = bert_score_fn(
        safe_preds, references,
        model_type="bert-base-multilingual-cased",
        batch_size=32, device="cpu", verbose=False,
    )
    bert_f1 = f1.mean().item()

    logger.info(
        "[%s] Метрики на %d примерах: ExactMatch=%.4f, BLEU=%.2f, chrF=%.4f, BERTScore F1=%.4f",
        label or "eval", n, exact_match, bleu, chrf, bert_f1,
    )

    return {"exact_match": exact_match, "bleu": bleu, "chrf": chrf, "bert_score_f1": bert_f1}
