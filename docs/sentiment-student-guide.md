# Трехклассовый студент: запуск и перенос

Студент принимает одно пользовательское сообщение и возвращает `NEGATIVE`, `NEUTRAL`
или `POSITIVE`, вероятности трех классов и `negative_score`. Он воспроизводит рубрику
зафиксированного Qwen. Объяснения учителя не нужны при предсказании.

## Установка на Windows, macOS или Linux

Из корня скопированного проекта создайте новое окружение Python 3.11+:

```text
python -m venv .venv
```

В Windows активируйте `.venv\Scripts\Activate.ps1`, в macOS/Linux —
`source .venv/bin/activate`. Если Python доступен как `python3`, используйте эту команду
при создании окружения. Старое Windows-окружение `.venv` переносить не нужно.

Для выбранной модели с энкодером:

```text
python -m pip install -e ".[ml]"
```

Для одного TF-IDF достаточно `python -m pip install -e ".[linear]"`.
Для воспроизведения всей серии, включая Parquet и графики:

```text
python -m pip install -e ".[ml,dataset,experiments]"
```

Переносите код проекта и `artifacts/sentiment-student-v1/`. Если выбран дообученный
энкодер, его веса включены в эту папку. Если выбран замороженный E5, также перенесите
`artifacts/hf/` либо разрешите разовую загрузку флагом `--allow-download`.
Загружается зафиксированная ревизия; по умолчанию сеть для предсказаний не используется.
Файлы данных, журнал разметки и другие кандидаты нужны только для воспроизведения
эксперимента. Каталоги `artifacts/`, `runs/`, `data/` исключены из Git: один clone
не переносит веса и результаты.

## Предсказания

Проверочный пример на подготовленных входах:

```text
python -m memory_trace predict-sentiment runs/sentiment-students-v1/data/test.jsonl --model-dir artifacts/sentiment-student-v1 --out runs/student-predictions.jsonl --device cpu
```

В выходном JSONL есть `input_id`, `sentiment_label`, `probabilities`, `negative_score`,
`model_version` и `mode: "shadow"`. Qwen для этого не запускается. `--device mps`
выбирает Apple GPU, `--device cuda` — NVIDIA, `--batch-size` управляет батчем.
Измеренные в исследовании скорости относятся к указанной в отчете машине.

Для новых сообщений можно подготовить входы через Python API:

```python
from memory_trace.sentiment import make_sentiment_input
from memory_trace.io import write_jsonl

inputs = [make_sentiment_input(
    "Please inspect this code.",
    tenant_id="my-work-log",
    session_id="session-001",
    message_id="message-001",
)]
write_jsonl("runs/my-messages.jsonl", inputs)
```

Одинаковые сообщения одной сессии должны иметь одинаковый `session_id`, а разные
события — разные `message_id`. Функция не придумывает историю ассистента; в этой серии
используется именно текст сообщения. Исходные классы для предсказания не требуются.

## Альтернативный бэкбон BGE-M3

Сравнение с E5 использует готовую разметку Qwen и отдельные папки результатов/весов.
Повторная разметка не нужна. Сначала загрузите конкретный checkpoint, затем запустите
автономный скрипт, который сам переключает этапы и пишет `progress.json` и логи:

```text
python scripts/cache_sentiment_encoder.py --base runs/sentiment-bge-m3-v1 --policy configs/student-study-bge-m3.json
python scripts/run_backbone_study.py --device cuda
```

Для временной остановки того же Docker-сервиса Qwen добавьте
`--qwen-compose-dir PATH`. Скрипт восстанавливает его в `finally`, в том числе при
ошибке обучения. Без этого аргумента управление Qwen не выполняется. На другой
машине используйте `--device cpu` или `--device mps` и подходящий PyTorch.

В сравнении окно BGE-M3 ограничено 512 токенами, как у E5, с сохранением всего текста
через окна. У BGE-M3 нет префикса `query: `, используется штатный CLS-пулинг.
Три эпохи, эффективный batch 32; микро-batch 2, accumulation 16 и gradient
checkpointing снижают расход VRAM. Сетка параметров головы и выбор эпохи используют
только calibration. Повторно используемый test из эксперимента E5 уже был просмотрен.

```text
python -m memory_trace predict-sentiment INPUT.jsonl --model-dir artifacts/sentiment-bge-m3-v1 --out predictions.jsonl --device cpu --batch-size 8
```

Префикс и размер окна при предсказании читаются из артефакта. Для **нового**
эксперимента `train-sentiment --kind encoder --model BAAI/bge-m3 --revision SHA`
выбирает без префикса и окно 8192 по умолчанию; `--max-length 512` воспроизводит
выбор текущего сравнения. Старые E5-артефакты сохраняют прежнее поведение.
Конфигурация исследования: `configs/student-study-bge-m3.json`;
[результат сравнения](experiments/sentiment-bge-m3-2026-09-15.md).

## Что означают пороги

`routing-policy.json` содержит пороги, подобранные только на calibration для целей
полноты 95%, 98% и 99%, и вероятности аудита 10%/20%. Это цели на calibration,
а не гарантия на новом потоке. Команда предсказаний только сохраняет оценки:
она не отбрасывает события и не отправляет их учителю автоматически.

В отчете проверяются два применения:

- Найти как можно больше `NEGATIVE`: отправлять Qwen сработавшие сессии и случайную
  долю остальных. Симуляция использует всю доступную сессию; поведение онлайн,
  когда следующие сообщения еще не пришли, отдельно не проверялось.
- Оценить долю `NEGATIVE`: сохранять вероятность включения каждой проверяемой сессии
  и использовать выборочные веса. Доля негативных среди одних срабатываний фильтра
  не оценивает частоту в общем потоке.

Для первых рабочих данных сохраняйте shadow-режим: оценки студента доступны для
анализа, а фактическая маршрутизация остается полной. Эта серия измеряет согласие
с автоматическим учителем; `NEGATIVE` не доказывает ошибку агента.

## Повторить эксперимент

Протокол и состав данных сохраняются до обучения. Скрипты ниже выполняются из корня
проекта и продолжают существующую серию; завершенная финальная проверка не используется
для повторного подбора параметров.

```text
python scripts/label_sentiment_students.py
python scripts/cache_sentiment_encoder.py
python scripts/embed_sentiment_students.py
python scripts/run_sentiment_study.py baselines
python scripts/run_sentiment_study.py finetune --device cpu
python scripts/run_sentiment_study.py select
python scripts/run_sentiment_study.py evaluate --device cpu
python scripts/run_sentiment_study.py report
```

Дообучение выполняется только при заранее зафиксированном условии на calibration.
Если доступна свободная NVIDIA GPU, для этого шага можно указать `--device cuda`.
Скрипт `complete_sentiment_study.py` выполняет заключительные этапы последовательно;
опциональный путь `--qwen-compose-dir` позволяет временно остановить сервис
`qwen38-llama` и запустить тот же контейнер после обучения. Конкретный путь машины
в код не зашит. При ручном запуске шагов память GPU освобождается вручную.

Подключение учителя задается в `configs/judge.local.json`; ключ — через переменную
окружения либо отдельный локальный файл. Для воспроизведения прежних меток нужны
тот же prompt, параметры и deployment, записанные в `teacher/run.json`.
