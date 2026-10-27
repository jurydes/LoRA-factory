import os
import random
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _no_model_catalog(monkeypatch):
    """Каталог моделей HF (сеть и LLM) в тестах не обновляется: сервис при старте запускает его фоном."""
    async def none(self, force=False):
        return 0
    monkeypatch.setattr("factory.orchestrator.Orchestrator.refresh_models", none)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Изолированный DATA_DIR с реестрами и заметками проекта."""
    d = tmp_path / "data"
    shutil.copytree(ROOT / "data" / "registry", d / "registry")
    if (ROOT / "data" / "docs").exists():
        shutil.copytree(ROOT / "data" / "docs", d / "docs")
    monkeypatch.setenv("FACTORY_DATA_DIR", str(d))
    from factory.core import config
    config.get_config.cache_clear()
    yield d
    config.get_config.cache_clear()


@pytest.fixture
def cfg(data_dir):
    from factory.core.config import get_config
    return get_config()


@pytest.fixture
def registry(cfg):
    from factory.core.registry import Registry
    return Registry.load(cfg)


@pytest.fixture
def cls_spec(registry):
    from factory.core.contracts import TaskSpec
    spec = TaskSpec(description="Классификация отзывов по тональности", task_type="sentiment_analysis",
                    language="ru", labels=["позитив", "негатив", "нейтрально"], base_model_id="Qwen/Qwen3-0.6B")
    return registry.apply_task_metrics(spec)


@pytest.fixture
def gen_spec(registry):
    from factory.core.contracts import TaskSpec
    spec = TaskSpec(description="Детоксификация комментариев", task_type="text_generation", language="ru",
                    base_model_id="Qwen/Qwen3-0.6B")
    return registry.apply_task_metrics(spec)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Llama с 2 слоями и BPE-токенизатором, обученным на кириллице, — сохраняется локально."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    words = ["хороший", "отличный", "прекрасный", "плохой", "ужасный", "отвратительный", "товар", "позитив", "негатив",
             "Input", "Target", "Ответь", "одной", "меткой", "из", "списка", "отзыв", "тональность"]
    corpus = [" ".join(random.Random(i).choices(words, k=12)) for i in range(400)]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    hf_tok.save_pretrained(d)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(hf_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=3))
    model.save_pretrained(d)
    return d


@pytest.fixture
def tiny_registered(data_dir, tiny_model):
    """Крошечная модель, добавленная в реестр изолированного DATA_DIR."""
    import yaml
    path = data_dir / "registry" / "models.yaml"
    models = yaml.safe_load(path.read_text(encoding="utf-8"))
    models.append({"id": str(tiny_model), "params_b": 0.0001, "hidden_size": 32, "num_layers": 2, "max_context": 256,
                   "languages": ["ru"], "license": "mit", "commercial_ok": True,
                   "lora_modules": {"attention": ["q_proj", "v_proj"], "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                   "unsloth": False, "ptuning": True})
    path.write_text(yaml.safe_dump(models, allow_unicode=True), encoding="utf-8")
    return tiny_model
