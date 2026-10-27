```mermaid
flowchart TD
    USER(["👤 Пользователь"])

    subgraph EXT ["Внешние системы"]
        HF[("HuggingFace Hub\nмодели + датасеты")]
        LLM[("Локальный LLM-сервер\nRuadaptQwen3-32B")]
        FS[("Файловая система\noutput_dir/")]
    end

    A["① Настройка\nмодель / адаптер (LoRA|P-tuning)\nописание задачи"]
    B["② Поиск и выбор датасета\n(LLM ранжирует по релевантности)"]
    C["③ Загрузка модели\n+ применение адаптера"]
    D["④ Метрики ДО обучения\n(ExactMatch, BERTScore)"]
    E["⑤ Подбор гиперпараметров\n(эвристики по датасету и метрикам)"]
    F["⑥ Обучение\n(HuggingFace Trainer)"]
    G["⑦ Метрики ПОСЛЕ + примеры\n(сравнение до/после, val_check)"]

    USER -- "выбор модели,\nадаптера, задачи" --> A
    B -- "топ-10 датасетов → выбор" --> USER
    E -- "предложенный конфиг\n+ обоснование" --> USER
    USER -- "принять или ввести вручную" --> E
    F -- "логи: loss, метрики\nна каждом eval-шаге" --> USER
    G -- "Input / Target / Prediction\nсравнение метрик" --> USER

    A --> B
    B -->|"scoring prompt"| LLM
    B -->|"датасет"| HF
    B --> C
    C -->|"модель"| HF
    C --> D
    D --> E
    E --> F
    F --> G
    G -->|"адаптер + конфиг"| FS
    F -->|"eval"| LLM

    classDef step fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef ext fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef user fill:#4a90d9,stroke:#2c5f8a,color:#fff,font-weight:bold

    class A,B,C,D,E,F,G step
    class HF,LLM,FS ext
    class USER user
```

## Точки обратного отклика с пользователем

| # | Когда | Что |
|---|-------|-----|
| ① | Старт | Выбор модели, адаптера, ввод описания задачи |
| ② | После поиска | Показ топ-10 датасетов с оценками → выбор id |
| ③ | После baseline | Предложенные гиперпараметры + объяснение → принять / ввести вручную |
| ④ | Во время обучения | Онлайн-логи: loss, ExactMatch, BERTScore на каждом eval-шаге |
| ⑤ | После обучения | Примеры Input/Target/Prediction + сравнение метрик до и после |
