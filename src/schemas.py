from dataclasses import dataclass, field
import torch


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    num_train_epochs: int = 25
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 1e-4
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    fp16: bool = True
    gradient_accumulation_steps: int = 4

    dataset_name: str = "banking77"
    max_length: int = 128
    val_split: float = 0.1

    output_dir: str = "./qwen3_lora"
    log_file: str = "./validation_results.txt"
    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_token: str = ""