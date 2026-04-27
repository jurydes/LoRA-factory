import json
import logging
import dataclasses
import numpy as np
from pathlib import Path
from settings import TrainConfig
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments
from bert_score import score as bert_score_fn

logger = logging.getLogger(__name__)


def _make_compute_metrics(tokenizer):
    def extract_target(text):
        return text.split("Target:", 1)[1].strip() if "Target:" in text else text.strip()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [extract_target(p) for p in decoded_preds]
        decoded_labels = [extract_target(l) for l in decoded_labels]

        exact_match = sum(p == r for p, r in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        _, _, f1 = bert_score_fn(decoded_preds, decoded_labels, model_type="bert-base-multilingual-cased", verbose=False)

        return {"exact_match": exact_match, "bert_score_f1": f1.mean().item()}

    return compute_metrics


def _preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def train_and_log(tokenizer, model, dataset, config: TrainConfig):
    """Запускает обучение через HuggingFace Trainer, логирует ExactMatch и BERTScore на каждом eval шаге.

    Сохраняет адаптер и токенизатор в config.output_dir. Возвращает обученную модель.
    """
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
        gradient_checkpointing=False,
        bf16=config.bf16,
        tf32=True,
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
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
    )
    trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    config_path = Path(config.output_dir) / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(config), f, ensure_ascii=False, indent=2)

    logger.info("Обучение завершено, модель сохранена в %s", config.output_dir)

    return model
