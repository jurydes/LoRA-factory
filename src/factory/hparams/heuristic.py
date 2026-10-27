"""Вариант A: эвристика по размеру датасета, размеру модели и метрике исходной модели до обучения.

Опора — практика LoRA (Unsloth LoRA Hyperparameters Guide; Thinking Machines «LoRA Without Regret», 2025):
lr LoRA ≈ 10× lr полного дообучения, почти не зависит от ранга и объёма данных — старт 2e-4; эпох 1–3;
адаптер на все линейные слои; эффективный батч ≤ 16. Прежние правила курсовой (config_gen.suggest_config)
снижали lr с ростом датасета до 5e-5 — на 25 тыс. примеров обучение шло вчетверо медленнее нужного."""
from __future__ import annotations

from factory.core.contracts import Evidence, ParamRationale
from factory.hparams.guard import HParamContext, LoraChoice, PtuningChoice, StrategyChoice

PTUNING_LR_FACTOR = 10.0
LORA_LR = 2e-4
LARGE_MODEL_HIDDEN = 4096  # у больших моделей lr полного дообучения ниже — и у LoRA тоже (×0.5)
GOOD_BASELINE, POOR_BASELINE = 0.75, 0.5

# (до скольких примеров, ранг, эпох): больше данных — больше ёмкость адаптера и меньше проходов.
# Эпохи — верхняя граница с ранним остановом, но по ним считаются оценка времени и бюджет GPU-часов.
SIZE_RULES = [(1_000, 8, 4), (10_000, 16, 3), (50_000, 32, 2), (float("inf"), 32, 1)]


def suggest(ctx: HParamContext, baseline_primary: float | None) -> StrategyChoice:
    """baseline_primary — основная метрика исходной модели с промптом на val (0–1), если посчитана."""
    _, r, epochs = next(rule for rule in SIZE_RULES if ctx.n_train < rule[0])
    lr = LORA_LR if ctx.model.hidden_size <= LARGE_MODEL_HIDDEN else LORA_LR / 2
    reason = (f"{ctx.n_train} примеров → r={r}, alpha={2 * r}, до {epochs} эп.; lr {lr:g} — стандарт LoRA, "
              f"не снижается с ростом данных" + ("" if lr == LORA_LR else f" (модель крупная: hidden {ctx.model.hidden_size})"))
    if baseline_primary is not None and baseline_primary > GOOD_BASELINE:
        epochs = max(1, epochs - 1)
        reason += f"; исходная модель уже справляется ({baseline_primary:.2f}) → эпох −1"
    elif baseline_primary is not None and baseline_primary < POOR_BASELINE and ctx.n_train < 10_000:
        epochs += 1
        reason += f"; исходная модель справляется плохо ({baseline_primary:.2f}) → эпох +1"
    evidence = [Evidence(kind="rule", ref="heuristic.SIZE_RULES"), Evidence(kind="doc", ref="lora.md")]
    rationales = [ParamRationale(param="lr", value=lr, reason=reason, evidence=evidence),
                  ParamRationale(param="epochs", value=epochs, reason=reason, evidence=evidence)]
    if ctx.adapter_kind == "lora":
        # все линейные слои и для классификации: LoRA только на attention проигрывает даже при равном числе
        # параметров (LoRA Without Regret), а основной вклад дают MLP-слои
        modules = "all_linear"
        return StrategyChoice(
            lr=lr, epochs=epochs,
            lora=LoraChoice(r=r, alpha=2 * r, modules=modules),
            rationales=rationales + [
                ParamRationale(param="lora.r/alpha", value=[r, 2 * r], reason=reason, evidence=evidence),
                ParamRationale(param="lora.modules", value=modules,
                               reason="все линейные слои (attention + MLP): только attention проигрывает при любом ранге",
                               evidence=[Evidence(kind="doc", ref="lora.md")])])
    lr *= PTUNING_LR_FACTOR
    rationales[0] = ParamRationale(param="lr", value=lr, evidence=evidence + [Evidence(kind="rule", ref="heuristic.PTUNING_LR_FACTOR")],
                                   reason=f"{reason}; P-tuning обучает только энкодер подсказок → lr ×{PTUNING_LR_FACTOR:g}")
    return StrategyChoice(
        lr=lr, epochs=epochs,
        ptuning=PtuningChoice(num_virtual_tokens=20, encoder_hidden_size=min(128, ctx.model.hidden_size)),
        rationales=rationales + [ParamRationale(param="ptuning", value="20 токенов, MLP 128",
                                                reason="значения по умолчанию курсовой (TrainConfig)",
                                                evidence=[Evidence(kind="rule", ref="settings.TrainConfig")])])
