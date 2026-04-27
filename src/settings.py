from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import SettingsConfigDict, BaseSettings
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field
import torch


PATH_TO_ENV = Path(__file__).parent / "dev.env"


class OpenAISettings(BaseModel):
    url: Union[str, HttpUrl] = Field(default=HttpUrl("http://localhost"))
    key: str = Field(default="EMPTY")
    default_model: str = Field(default="RefalMachine/RuadaptQwen3-32B-Instruct-v2")


class ModuleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PATH_TO_ENV.as_posix(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter='__',
        nested_model_default_partial_update=True,
        protected_namespaces=(),
    )

    openai_server: OpenAISettings = OpenAISettings()
    hf_token: str = Field(default="")


def load_global_config():
    return ModuleSettings()


@dataclass
class TrainConfig:
    model_name: str = "google/gemma-2b"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    adapter_type: str = "lora"  # "lora" или "p-tuning"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj", "k_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    num_virtual_tokens: int = 20
    encoder_hidden_size: int = 128

    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 1e-3
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 400
    save_steps: int = 50
    save_total_limit: int = 2
    bf16: bool = True if torch.cuda.is_bf16_supported() else False
    gradient_accumulation_steps: int = 4

    dataset_name: str = "banking77"
    max_length: int = 128
    max_new_tokens: int = 128
    val_split: float = 0.1
    target_count: int = 2000

    output_dir: str = "./qwen3_lora"
    log_file: str = "./validation_results.txt"
    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_token: str = ""

    num_val_samples: int = 5
    num_eval_samples: int = 50


HYPERPARAMETER_PRESETS = {
    "1": {
        "name": "Быстрый (отладка)",
        "description": "2 эпохи, lr=1e-2 — быстрая проверка что всё работает",
        "params": dict(num_train_epochs=2, learning_rate=3e-4, lora_r=8, lora_alpha=16),
    },
    "2": {
        "name": "Базовый",
        "description": "4 эпохи, lr=3e-3 — хороший старт",
        "params": dict(num_train_epochs=4, learning_rate=1e-4, lora_r=8, lora_alpha=16),
    },
    "3": {
        "name": "Глубокий",
        "description": "8 эпох, lr=2e-4, r=16 — для сложных задач",
        "params": dict(num_train_epochs=8, learning_rate=5e-5, lora_r=16, lora_alpha=32),
    },
    "4": {
        "name": "Своя конфигурация",
        "description": "Ввести параметры вручную",
        "params": {},
    },
}
