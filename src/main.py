from src.schemas import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import multiprocessing
import traceback
from load_model import load_model
from load_dataset import create_dataset
from train import train_and_log
from inference import val_check

def _training_worker(config: TrainConfig):
    """
    Изолированная функция. 
    Все объекты (модель, токенизатор, тензоры) создаются здесь и уничтожаются вместе с процессом
    """
    try:
        tokenizer, model = load_model(config)
        raw, dataset = create_dataset(config, tokenizer)

        trained = train_and_log(tokenizer, model, dataset, config)
        
        # Запуск инференса для проверки
        val_check(tokenizer, trained, raw["test"], config)
        
    except Exception as e:
        print(f"Критическая ошибка в процессе обучения: {e}")
        traceback.print_exc()
        raise SystemExit(1)

def run_lora_pipeline(config: TrainConfig):
    """
    Запуск пайплайна в изолированном процессе
    """
    # Используем 'spawn' для безопасной работы с CUDA-контекстом
    ctx = multiprocessing.get_context('spawn')
    
    # Создаем и запускаем дочерний процесс
    process = ctx.Process(target=_training_worker, args=(config,))
    process.start()
    
    # Блокируем главный поток, пока дочерний процесс не завершит работу (и не освободит память)
    process.join()

    # Проверяем код возврата (0 = успешно)
    if process.exitcode == 0:
        return f"Модель {config.model_name} успешно обучена, проверена и сохранена в {config.output_dir}"
    else:
        raise RuntimeError(f"Процесс обучения завершился аварийно с кодом: {process.exitcode}")

# Пример точки входа (если запускать main.py напрямую)
if __name__ == "__main__":
    config = TrainConfig()
    result_message = run_lora_pipeline(config)
    print(result_message)

# Датасет подгрузили
    # Начинаем обучать
    # [450/450 10:52, Epoch 25/25]
    # Step	Training Loss	Validation Loss
    # 50	2.080200	2.123073
    # 100	1.737468	1.909117
    # 150	1.532364	1.808899
    # 200	1.301868	1.748557
    # 250	1.170546	1.716241
    # 300	1.053347	1.716059
    # 350	0.981155	1.716900
    # 400	0.927908	1.730666
    # 450	0.900962	1.740499

# Обучили, проверяем
    # [1]
    # Input:      javascript book-series training-materials ES6 closures prototypes async
    # Target:     book-series, javascript, closures, prototypes, async, es6, es2015, training-materials, book, training-providers
    # Prediction:  javascript, book, book-series, training-materials, learning, learning-program, training, resources, es6, prototypes, async-await, closures, promises, javascript-books, javascript-learning, free-pdf, es6-training, pdf, javascript-training, javascript-books-list, hobbes, hobbes-javascript, free

    # [2]
    # Input:      javascript snippets" or "nodejs snippets" or "css snippets" in awesome-list or learning-resources or learn-to-code or education
    # Target:     awesome-list, javascript, snippets, learning-resources, learn-to-code, programming, education, es6-javascript, nodejs, css
    # Prediction:  javascript, nodejs, css, snippets, lovebanned, awesome-list, learning-resources, education, programming, code-competitions, codebase-attacks, css-in-js, regex, promise, classnames, async-await, document-strings, modular-css, postcss, prettier, vite, vite

    # [3]
    # Input:      axios" or "node-fetch" or "got" or "unfetch" or "superagent
    # Target:     http-client, javascript, nodejs, promise, hacktoberfest
    # Prediction:  axios, node-fetch, got, unfetch, got, node-fetch, superagent, axios, http, promise, url, urljs, urlmagic, urlparser, urlparserjs, urlparser-node, urlmagicjs, urlmagic-node, nodejs, javascript, javascript-http, javascript-http-client, javascript

    # [4]
    # Input:      Need an AI system that can redact personal information from documents automatically
    # Target:     pii-redaction, nlp, document-ai, data-privacy, anonymization, text-processing
    # Prediction:  document-redaction, personal-data-encryption, ai-automation, nlp, data-security, text-redaction, text-generation, generative-ai, generative-models, generative-framework, generative-frameworks, ai-tools, ai-resources, ai-prank, pranks, bad-ai, bad-

    # [5]
    # Input:      Awesome curated lists for web development in 2025
    # Target:     awesome-list, web-development, javascript, curated, 2025
    # Prediction:  web-development, 2025, collections, lists, awesome, programming, programming-resources, programming-books, react, angular, pwa, javascript, css, html, d3, nodejs, python, go, go-language, golang, viper, python-language, shell, npm, eriche
