# Assistive Gait Research

Проект изучает распознавание условий ходьбы по данным инерциальных датчиков NTNU. Один объект для модели — цикл шага. Классы заданы числовыми кодами `Terrain`; названия кодов 8 и 9 остаются спорными. Модель служит для учебного исследования и демонстрации лабораторной №10.

Исходные CSV и результаты находятся рядом с репозиторием в `../research-publication-data/`. Исходные файлы не изменяются и не входят в Git. Описание набора и замеченные проблемы разметки — в [документе о данных](docs/dataset.md).

## Установка

Нужны Python 3.12 и [uv](https://docs.astral.sh/uv/getting-started/installation/). Из корня проекта:

```powershell
uv sync --locked --all-groups
```

Если PowerShell не находит `uv`, проверьте его установку командой `python -m pip show uv` и добавьте папку с `uv.exe` в `PATH`.

## Исследование

Запускайте ноутбуки по порядку в окружении проекта:

1. [01 — EDA](notebooks/01_ntnu_eda_simple.ipynb): состав файлов, качество данных и распределение классов.
2. [02 — воспроизведение Wassall et al. (2026)](notebooks/02_reproduce_wassall_2026.ipynb): признаки и сравнение CV, LSO, LOPO.
3. [03 — эксперименты](notebooks/03_model_experiments.ipynb): очистка циклов, длина представления и каналы датчиков.
4. [04 — проверка модели](notebooks/04_model_validation.ipynb): нормализация, сравнение алгоритмов и независимая оценка.

Папка данных: `../research-publication-data/data/raw/ntnu/`. Ноутбуки сохраняют таблицы и рисунки в `../research-publication-data/artifacts/reports/`. Функции фильтрации и расчёта признаков находятся в `src/research_graph/pipelines/`, пути — в `src/research_graph/paths.py`. Разделение по участникам и параметры экспериментов видны в самих ноутбуках.

Краткие итоги и ограничения модели собраны в [отчёте лабораторной №10](docs/lab10_report.md). На четырёх отложенных участниках macro-F1 выбранной модели составил 0,299. Рабочий артефакт, обученный на всех 20 участниках, не является независимой оценкой.

## API и интерфейс

Лабораторный пример с ипотекой адаптирован к циклам шага NTNU. Сервис использует Random Forest без каденса, с 140 признаками. Артефакты лежат в `../research-publication-data/artifacts/reports/model_validation/`. Повторно подготовить их можно командой:

```powershell
uv run --locked --all-groups python scripts/prepare_lab10_artifacts.py
```

Запуск API:

```powershell
uv run --locked --all-groups python -m uvicorn research_graph.api.main:app --host 127.0.0.1 --port 8000
```

В другом терминале:

```powershell
cd frontend
npm ci
npm run dev
```

Откройте `http://127.0.0.1:5173`. Загрузите доверенный `terrain_predictor_final.joblib`, затем PS CSV из папки NTNU или CSV с готовыми признаками. Также доступен ручной ввод признаков. Ответ содержит класс и вероятности. При наличии меток `Terrain` сервис показывает метрики; ROC-AUC рассчитывается только при наличии всех классов.

Маршруты: `POST /upload-model`, `POST /predict`, `POST /predict-from-csv`, `GET /model-info`. Примеры запросов и ограничения загрузки описаны в [отчёте](docs/lab10_report.md); интерактивная схема доступна по `http://127.0.0.1:8000/docs`. Список доверенных контрольных сумм — в `src/research_graph/api/trusted_models.json`. Переобученный файл нужно проверять перед добавлением в этот список.

## Проверки

```powershell
uv run --locked --all-groups ruff check .
uv run --locked --all-groups ruff format --check .
uv run --locked --all-groups mypy
uv run --locked --all-groups pytest
cd frontend
npm run lint
npm run test
npm run build
```

CI настроен в `.sourcecraft/ci.yaml` и `.github/workflows/ci.yml`. Промпты для отчёта по лабораторной сохранены в [отдельном файле](docs/lab10_ai_prompts.md). Git-репозиторий связан с [debikde/python-programming](https://github.com/debikde/python-programming), но локальные изменения не отправляются автоматически.
