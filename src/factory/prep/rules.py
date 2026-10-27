"""Статистические правила: нужна ли чистка и вызывать ли агента A2."""
from __future__ import annotations

from dataclasses import dataclass, field

from factory.core.config import PrepRules
from factory.core.contracts import DataProfile, DedupeOp, FilterOp, RecipeOp, TaskFamily

SMALL_DATASET = 300


@dataclass
class RuleVerdict:
    call_agent: bool
    findings: list[str] = field(default_factory=list)
    ops: list[RecipeOp] = field(default_factory=list)


def assess(profile: DataProfile, family: TaskFamily, rules: PrepRules, language: str) -> RuleVerdict:
    """Однозначные проблемы (пустые, точные дубли) чинятся операциями-правилами без LLM.
    Неоднозначные (дисбаланс, смесь языков, мало данных, очень длинные ответы) — повод вызвать агента."""
    v = RuleVerdict(call_agent=False)
    n = max(profile.n_rows, 1)

    if profile.n_empty:
        v.ops.append(FilterOp(field="input_len", operator=">", value=0))
        if family == "generation" or profile.n_empty / n > rules.max_empty_share:
            v.ops.append(FilterOp(field="target_len", operator=">", value=0))
        v.findings.append(f"пустых строк: {profile.n_empty} ({profile.n_empty / n:.1%})")
    if profile.n_exact_dupes:
        v.ops.append(DedupeOp(mode="exact"))
        v.findings.append(f"точных дубликатов: {profile.n_exact_dupes} ({profile.n_exact_dupes / n:.1%})")
        if profile.n_exact_dupes / n > rules.max_dupe_share:
            v.call_agent = True
            v.findings.append("доля дублей велика — возможны и near-дубли")

    if family == "classification" and profile.imbalance_ratio and profile.imbalance_ratio > rules.max_imbalance:
        v.call_agent = True
        v.findings.append(f"дисбаланс классов {profile.imbalance_ratio:.1f}× (порог {rules.max_imbalance}×)")
    share = profile.language_share.get(language, 0.0)
    if profile.language_share and share < rules.min_lang_share:
        v.call_agent = True
        v.findings.append(f"доля языка {language}: {share:.0%} (порог {rules.min_lang_share:.0%})")
    if profile.target_tokens_est.p99 > rules.max_target_tokens_p99:
        v.call_agent = True
        v.findings.append(f"очень длинные ответы: p99 ≈ {profile.target_tokens_est.p99:.0f} токенов")
    if profile.n_rows < SMALL_DATASET:
        v.call_agent = True
        v.findings.append(f"мало данных: {profile.n_rows} строк в train")
    return v
