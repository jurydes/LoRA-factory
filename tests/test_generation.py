"""Генерация для оценки: остановка на первой строке, порядок ответов при сортировке по длине, батч по памяти."""
import torch

from factory.training.evaluator import FirstLineStop, generate


class _Tok:
    """Словарь из пяти токенов: текст, перевод строки, пробел, «.\n», пустые строки."""
    vocab = ["Ответ", "\n", " ", ".\n", "\n\n"]

    def __len__(self):
        return len(self.vocab)

    def batch_decode(self, ids):
        return [self.vocab[i[0]] for i in ids]


def test_first_line_stop_waits_for_text_then_newline():
    stop = FirstLineStop(_Tok(), vocab=8, batch=3, device="cpu")
    steps = [[1, 3, 4], [0, 0, 4], [2, 0, 4], [1, 0, 4]]  # по столбцам: «\nОтвет \n», «.\n», «\n\n\n\n…»
    done = []
    for k in range(len(steps)):
        ids = torch.tensor(steps[:k + 1]).T
        done.append(stop(ids, None).tolist())
    # A: ведущий перевод строки не останавливает, перевод строки после текста — да; B: «.\n» — сразу;
    # C: одни пустые строки — не останавливает
    assert [d[0] for d in done] == [False, False, False, True]
    assert done[0][1] is True and not any(d[2] for d in done)


def test_generate_keeps_order_with_length_sorting(tiny_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_model)
    model = AutoModelForCausalLM.from_pretrained(tiny_model)
    prompts = ["отзыв", "отзыв хороший товар отличный прекрасный товар", "плохой", "товар товар"]
    one = [generate(model, tok, [p], max_new_tokens=6, batch_size=1)[0] for p in prompts]
    assert generate(model, tok, prompts, max_new_tokens=6, batch_size=3) == one
    assert generate(model, tok, prompts, max_new_tokens=6) == one  # батч по памяти


def test_unsloth_generation_buffers_are_released():
    """Буферы KV-кэша быстрой генерации Unsloth удаляются сразу после оценки, а не в первом шаге обучения."""
    import torch

    from factory.training.evaluator import release_unsloth_buffers

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    attn = model[0]
    attn.paged_attention = torch.zeros(4)
    attn.paged_attention_K, attn.temp_QA, attn.attention = attn.paged_attention[:2], torch.zeros(2), torch.zeros(3)
    attn.scalar = 0.5
    release_unsloth_buffers(model)
    assert not any(hasattr(attn, n) for n in ("paged_attention", "paged_attention_K", "temp_QA", "attention"))
    assert attn.scalar == 0.5 and isinstance(attn.weight, torch.nn.Parameter)
