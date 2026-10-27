"""Формат примеров для SFT и генерации.

Без шаблона чата — формат курсовой: «{инструкция}\n\nInput: …\nTarget:» + « ответ».
С шаблоном чата (registry/models.yaml: chat_template) — то же сообщение пользователя через tokenizer.apply_chat_template;
промпт строится одинаково на обучении и при оценке."""
from __future__ import annotations

from typing import Callable

from factory.core.contracts import TaskSpec
from factory.core.registry import TaskInfo

PromptFn = Callable[[str], str]


def instruction(spec: TaskSpec, task: TaskInfo) -> str:
    """Инструкция под формат ответа задачи."""
    if task.format == "label":
        return f"{spec.description}\nОтветь одной меткой из списка: {', '.join(spec.labels or [])}."
    if task.format == "entities":
        types = f" типов: {', '.join(spec.labels)}" if spec.labels else ""
        return (f"{spec.description}\nВыдели именованные сущности{types}. Ответь одной строкой в формате "
                "«ТИП: текст; ТИП: текст» в порядке появления в тексте; если сущностей нет — «нет».")
    if task.format == "span":
        return f"{spec.description}\nОтветь на вопрос коротким фрагментом из контекста, без пояснений."
    return f"{spec.description}\nОтвет — одной строкой."


def prompt_builder(instr: str, tokenizer=None, chat: bool = False) -> PromptFn:
    def plain(inp: str) -> str:
        return f"{instr}\n\nInput: {inp}\nTarget:"

    if not chat or tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return plain

    def chat_prompt(inp: str) -> str:
        messages = [{"role": "user", "content": f"{instr}\n\nInput: {inp}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return chat_prompt


def build_prompt(instr: str, inp: str) -> str:
    return prompt_builder(instr)(inp)


def completion(target: str) -> str:
    return f" {target}"


def to_sft_records(inputs: list[str], targets: list[str], prompt: PromptFn) -> list[dict]:
    return [{"prompt": prompt(i), "completion": completion(t)} for i, t in zip(inputs, targets)]


def extract_answer(generated: str) -> str:
    """Первая непустая строка ответа (как inference._generate_prediction курсовой)."""
    for line in generated.split("\n"):
        if line.strip():
            return line.strip()
    return ""
