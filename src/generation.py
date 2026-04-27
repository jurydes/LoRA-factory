from async_lru import alru_cache
from openai import AsyncOpenAI, OpenAI
from tqdm.auto import tqdm
from settings import load_global_config
import asyncio
import inspect

# ДЛЯ БЕЗОПАСНОСТИ ЭТО НУЖНО ОФОРМИТЬ КАК ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# И ПОДГРУЖАТЬ У СЕБЯ НА МАШИНЕ ФАЙЛ С РАЗРЕШЕНИЕМ .ENV
# !!! ФАЙЛ .ENV должен быть в .gitignore (чтобы он не попал в сеть)

config = load_global_config().openai_server
API_KEY=config.key
API_URL = config.url
DEFAULT_MODEL = config.default_model


def get_client():
    return OpenAI(api_key=API_KEY, base_url=API_URL)


def get_asynclient():
    return AsyncOpenAI(api_key=API_KEY, base_url=API_URL, timeout=60.0)


def get_available_models():
    tmp_client = get_client()

    model_list = [f"{mod.id}" for mod in tmp_client.models.list().data]
    return model_list


def separate_thinking(text):
    if "</think>" not in text:
        return text
    return "".join(text.split("</think>")[1:]).strip()


@alru_cache(maxsize=8192)
async def generate_task(
    prompt, model_name=None,
    temperature=0.1, top_p=0.9, max_tokens=2000,
    rep_penalty=1.0, post_process=True,
    freq_penalty=0.0, min_p=0.2, use_thinking=False,
    answer_prefix=None
):
    oclient = get_asynclient()
    if model_name is None:
        model_name = DEFAULT_MODEL
    messages = [{"role": "user", "content": prompt}]
    if answer_prefix:
        messages.append({"role": "assistant", "content": answer_prefix})
    res = await oclient.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        frequency_penalty=freq_penalty,
        extra_body={
            "repetition_penalty": rep_penalty,
            "min_p": min_p,
            "add_generation_prompt": len(messages) == 1,
            "chat_template_kwargs": {
                "enable_thinking": use_thinking
            },
            "top_k": 100,
            "continue_final_message": len(messages) > 1
        }
    )
    try:
        # print(res)
        res = res.choices[0].message.content
        # print(prompt, res, sep='\n\n')
        if post_process:
            res = separate_thinking(res)
    except Exception as e:
        print(e)
        return res
    return res


# Класс для эффективной обработки нескольких одновременных запросов к API
class AsyncList:
    def __init__(self):
        self.contents = []
        self.couroutine_ids = []

    def append(self, item):
        self.contents.append(item)
        if inspect.iscoroutine(item):
            self.couroutine_ids.append(len(self.contents) - 1)

    async def complete_couroutines(self, batch_size=10, verbose=False):
        tracker = tqdm(total=len(self.couroutine_ids), disable=not verbose)
        while len(self.couroutine_ids) > 0:
            tasks = [self.contents[i] for i in self.couroutine_ids[:batch_size]]
            res = await asyncio.gather(*tasks)
            for i, r in zip(self.couroutine_ids, res):
                self.contents[i] = r
            self.couroutine_ids = self.couroutine_ids[batch_size:]
            tracker.update(len(tasks))

    def __getitem__(self, key):
        return self.contents[key]

    def __repr__(self):
        return repr(self.contents)

    def __len__(self):
        return len(self.contents)

    async def to_list(self):
        await self.complete_couroutines(batch_size=1)
        return self.contents
