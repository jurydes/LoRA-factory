import logging
from settings import TrainConfig, load_global_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PromptEncoderConfig, TaskType, get_peft_model
from huggingface_hub import login
import torch

logger = logging.getLogger(__name__)


def load_model(config: TrainConfig):
    """Загружает токенизатор и модель с применённым адаптером (LoRA или P-tuning).

    Если в конфиге указан HF токен, выполняет login для доступа к закрытым моделям.
    Возвращает (tokenizer, peft_model).
    """
    hf_token = load_global_config().hf_token
    if hf_token:
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16 if config.bf16 else "auto",
        device_map={"": config.device}
    )

    if config.adapter_type == "p-tuning":
        adapter_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_hidden_size=config.encoder_hidden_size,
        )
        logger.info(
            "Модель %s загружена, P-tuning применён (virtual_tokens=%d, device=%s)",
            config.model_name, config.num_virtual_tokens, config.device,
        )
    else:
        adapter_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type,
        )
        logger.info(
            "Модель %s загружена, LoRA применена (r=%d, alpha=%d, device=%s)",
            config.model_name, config.lora_r, config.lora_alpha, config.device,
        )

    return tokenizer, get_peft_model(model, adapter_config)
