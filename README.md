# Assistive Gait Research

[![CI](https://github.com/debikde/python-programming/actions/workflows/ci.yml/badge.svg)](https://github.com/debikde/python-programming/actions/workflows/ci.yml)

Исследовательский проект для анализа двигательной активности пользователей
протезов нижних конечностей по данным носимых инерциальных датчиков (IMU).
Задача ML — классификация условий ходьбы по циклам шага.
Планируемый источник: **IMU data from lower-limb prosthetic users traversing
real-world terrain** (NTNU), [DOI 10.18710/U8RGDL](https://doi.org/10.18710/U8RGDL).
Это исследовательский прототип, а не клиническая система диагностики.

## Текущий этап: лабораторная №10 на данных NTNU

Реализованы устанавливаемый пакет `research_graph`, CLI `rg`, конфигурация,
логирование, изолированные тесты, строгая типизация, pre-commit и CI.
Название Python-дистрибутива — `assistive-gait-research`; адрес существующего
GitHub-репозитория — `debikde/python-programming`. Переименование удалённого
репозитория для работы пакета не требуется.

Добавлены исходные данные всех 20 участников: полный архив дополнил прежние
две загрузки файлами P18–P20. Текущий EDA-ноутбук уже прочитал все 2552 CSV;
старый HTML-отчёт охватывает 2476 CSV участников P1–P17.
P7_TT и P7_TF — два представления одного человека.
Предыдущий отчёт выявил дубликаты OS/TR,
ошибки имён, Terrain=0/99, спорные коды 8/9, Turn=2 и масштаб P16.
Сравнение нормализации, Random Forest, Logistic Regression и Extra Trees
по группам новых участников выполнено в отдельном ноутбуке. Модель для
лабораторной №10 выбрана на 16 участниках и проверена на четырёх отложенных;
macro-F1 на них составил 0,299. Прежний HTML-отчёт EDA полной когорты
ещё не обновлялся.
FastAPI и интерфейс Vue 3/TypeScript реализованы для лабораторной №10.
Научный сайт остаётся отдельным следующим этапом.
Бейдж отражает удалённый workflow; текущие локальные изменения ещё не опубликованы.

## Окружение и установка

Требуются Git, uv и **Python 3.12** (версия указана в `.python-version`).
Стек выполнения: Pydantic, pydantic-settings, Typer.
Инструменты разработки: Ruff, mypy, pytest, pytest-cov, pre-commit,
pre-commit-hooks и httpx. FastAPI, Uvicorn, NumPy, pandas, SciPy,
scikit-learn и joblib входят в основное окружение. Группа `eda` добавляет
Jupyter и инструменты исследовательских ноутбуков.

Установите uv по [официальной инструкции](https://docs.astral.sh/uv/getting-started/installation/)
и убедитесь, что каталог с `uv` доступен в `PATH`:

```powershell
uv --version
uv python install 3.12
```

В текущей локальной папке проекта:

```powershell
uv sync --locked --all-groups
```

`uv sync` также поддерживается. Для воспроизводимой установки используйте
`--locked`: несовпадение `pyproject.toml` и `uv.lock` станет ошибкой.
Если `uv` не найден в PowerShell, в текущей Windows-среде он установлен как
`C:\Users\Zina\AppData\Roaming\Python\Python312\Scripts\uv.exe`;
его можно вызвать через `& "...\uv.exe"` или добавить каталог в `PATH`.
Первичная установка требует интернета; сами тесты работают без сети.
После будущей публикации можно будет клонировать репозиторий:

```powershell
git clone https://github.com/debikde/python-programming.git
cd python-programming
uv sync --locked
```

## Данные и EDA

Анализ также оформлен как [ноутбук-отчёт](notebooks/01_ntnu_eda.ipynb)
с пояснениями, таблицами и кодом 11 графиков. В текущем файле сохранены
результаты выполненных ячеек. Часть текстовых выводов относится к предыдущему
EDA по 17 людям и обозначена отдельно.

[Воспроизведение Wassall et al. (2026)](notebooks/02_reproduce_wassall_2026.ipynb)
оформлено как отдельный выполненный ноутбук. Он сравнивает CV, LSO и LOPO
с опубликованными результатами на уже имеющихся CSV.
[Собственные эксперименты E1–E4](notebooks/03_model_experiments.ipynb)
проверяют очистку циклов, представление сигнала и состав датчиков на
одинаковых группах новых участников; результаты и ограничения показаны в ноутбуке.
[Проверка моделей для лабораторной №10](notebooks/04_model_validation.ipynb)
сравнивает способы нормализации, алгоритмы и вклад каналов. Обученный объект
с предобработкой и таблицы сохранены локально в
`../research-publication-data/artifacts/reports/model_validation/`.
Низкое качество на отложенных людях не позволяет заявлять о готовности модели
к практическому применению.

## Лабораторная №10: сервис и демонстрация

Лабораторный пример с ипотекой адаптирован к NTNU IMU: одна запись ответа —
**цикл шага**, а класс — числовой код `Terrain`. [Методика, сравнительная
таблица и ограничения](docs/lab10_report.md) и
[зафиксированная оценка](docs/lab10_validation.json) хранятся в репозитории.
Экспериментальная модель обучена на 16 участниках и проверена на четырёх.
Отдельная рабочая модель обучена на всех 20; её нельзя оценивать метриками
экспериментальной копии как результат независимого теста.

Модели находятся вне Git, в соседнем каталоге
`../research-publication-data/artifacts/reports/model_validation/`:
`terrain_predictor.joblib` и `terrain_predictor_final.joblib`. Чтобы
воспроизвести рабочую копию на той же версии данных, выполните
`uv run --locked --all-groups python scripts/prepare_lab10_artifacts.py`.
Контрольные суммы разрешённых файлов закреплены в
`src/research_graph/api/trusted_models.json`; если артефакт пересоздан с
другой суммой, обновите реестр только после проверки его происхождения.
Произвольные загруженные pickle/joblib-файлы не десериализуются.

Запуск API из корня проекта:

```powershell
uv sync --locked --all-groups
uv run --locked --all-groups python -m uvicorn research_graph.api.main:app --host 127.0.0.1 --port 8000
```

Во втором терминале запустите интерфейс:

```powershell
cd frontend
npm ci
npm run dev
```

Откройте `http://127.0.0.1:5173`. Загрузите один из указанных файлов модели,
затем PS CSV из `../research-publication-data/data/raw/ntnu/P1/` с выбранной
стороной протеза либо CSV с 140 именованными признаками. Доступен шаблон
признаков и ручной ввод JSON-массива из 140 чисел. Ответ содержит вероятности
восьми классов; метрики выводятся при наличии допустимого `Terrain`.
Если тестовый CSV содержит не все восемь классов, ROC-AUC OvR не определён.
OpenAPI доступен по `http://127.0.0.1:8000/docs`.

`POST /upload-model` принимает только точную копию зарегистрированного файла;
`POST /predict` принимает `{"records": [[...140 чисел...]]}`;
`POST /predict-from-csv` принимает multipart-поля `file` и `side` (`left` или
`right` для исходного PS CSV). `GET /model-info` сообщает схему и результаты.
До загрузки модели маршруты предсказания отвечают HTTP 409.

Локальные проверки: `uv run --locked --all-groups pytest`, `ruff check .`,
`ruff format --check .`, `mypy`; во `frontend` — `npm run lint`,
`npm run test`, `npm run build`, `npm audit --audit-level=high`.
Такие же команды заданы в [SourceCraft CI](.sourcecraft/ci.yaml) и GitHub Actions.
Промпты использования ИИ для отчёта сохранены в
[docs/lab10_ai_prompts.md](docs/lab10_ai_prompts.md).

```powershell
uv sync --locked --group eda
uv run --locked --group eda python scripts/import_ntnu.py "C:\Users\Zina\Downloads\32756211.zip"
uv run --locked --group eda python scripts/eda_ntnu.py
```

Исходные байты и контрольные суммы в текущей рабочей среде:
`../research-publication-data/data/raw/ntnu/`. Ноутбук воспроизведения также
ищет набор в `data/raw/ntnu/` внутри проекта.
QC-реестр: `data/processed/ntnu/trial_registry.csv`.
После отдельного запуска скрипта графики PNG/SVG и таблицы появятся в
`artifacts/reports/eda_ntnu/`. Текущий HTML и текст `docs/eda_report.md`
описывают прежние 17 участников; для нового HTML сначала обновите выводы в тексте.
Данные и результаты исключены из Git; после клонирования нужен локальный архив.

[Полный EDA и следующие шаги](docs/eda_report.md) ·
[Описание схемы и источника](docs/dataset.md) ·
[Локальный HTML-отчёт](artifacts/reports/eda_ntnu/report.html).

Сводки рельефа считаются один раз по PS, с исключением одной полностью
дублирующей записи. Исходники не исправлялись; неоднозначные метки сохранены.
Сохранённые результаты относятся к прежним 17 участникам; исходные данные
теперь содержат всю когорту из 20 человек.
Оба представления P7 должны оставаться в одной группе при разделении выборки.
Длительности и частоты условны относительно документированных 100 Гц.

## Конфигурация

Настройки читаются из `RG_*` и необязательного `.env` в текущей рабочей папке.
Переменные окружения имеют приоритет над `.env`, затем используются значения
по умолчанию. Отсутствие `.env` и токена не мешает запуску.
По желанию создайте локальный файл: `Copy-Item .env.example .env` (PowerShell)
или `cp .env.example .env` (Linux/macOS). `.env` исключён из Git.

| Переменная | По умолчанию | Назначение и ограничения |
| --- | --- | --- |
| `RG_GITHUB_TOKEN` | пусто / `None` | Необязательный токен будущей интеграции; выводится только признак наличия |
| `RG_DATA_DIR` | `data` | Каталог данных; относительный путь считается от рабочей папки |
| `RG_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `NOTSET`; регистр не важен |
| `RG_REQUEST_TIMEOUT` | `10.0` | Конечное число секунд от 0.1 до 300 включительно |
| `RG_MAX_CONCURRENCY` | `4` | Целое число от 1 до 64 включительно |

Загрузка `Settings` не создаёт каталоги. `check-config` вызывает
`ensure_data_dir()`: создаёт недостающие родительские каталоги, сохраняет
существующее содержимое и сообщает об ошибке, если путь занят файлом
или недоступен для записи. `get_settings()` кэширует настройки в процессе;
после изменения окружения в интерактивной сессии вызовите
`get_settings.cache_clear()`.

## CLI

```powershell
uv run rg --help
uv run rg version
uv run rg check-config
```

`version` выводит `0.1.0`. При настройках по умолчанию `check-config` выводит:

```text
data_dir=data
log_level=INFO
request_timeout=10.0
max_concurrency=4
github_token_present=false
```

При заданном токене последняя строка содержит `true`; значение токена
не выводится. Неверная конфигурация или ошибка создания каталога даёт код
завершения 2 и краткое сообщение в stderr без значений настроек.
Логи идут в stderr в формате `время уровень имя_логгера: сообщение`.
Повторная настройка заменяет обработчики; шум сторонних библиотек ограничен.
Команд анализа данных на этапе ЛР1 нет.

## Проверки качества

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files
```

mypy работает в строгом режиме для `src` и `tests`. pytest использует
`--strict-markers`, branch coverage и порог 70%. Тесты API создают небольшой
синтетический доверенный артефакт, поэтому проходят и без локального архива
NTNU; дополнительные проверки настоящей модели выполняются, если он есть.

Хуки выполняют Ruff, mypy, проверку YAML, пробелов, окончаний строк,
конфликтов слияния и размера файлов. Через `uv run --locked` они используют
то же окружение и версии, что основной проект. После добавления новых файлов
в Git выполните pre-commit снова: `--all-files` проверяет отслеживаемые файлы.
Для автоматического запуска при локальном коммите: `uv run pre-commit install`.

GitHub Actions запускает проверки на push и pull request с Python 3.12
и `uv sync --locked --all-groups`. Подход к установке описан в
[документации uv для GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/).
Результаты локальной проверки — в [отчёте ЛР1](docs/lab1_report.md).

## Структура

```text
.
├── src/research_graph/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging_setup.py
│   ├── sources/{__init__.py,ntnu_loader.py}
│   ├── models/{schemas.py,terrain_predictor.py}
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── storage/{__init__.py,artifacts.py}
│   └── api/{main.py,routes.py,inference.py,trusted_models.json}
├── frontend/{src/,package.json,package-lock.json,vite.config.ts}
├── website/{README.md,docs/index.md}
├── notebooks/{01_ntnu_eda.ipynb,02_reproduce_wassall_2026.ipynb,
│             03_model_experiments.ipynb,04_model_validation.ipynb}
├── data/{raw,processed}/.gitkeep
├── artifacts/{models,reports}/.gitkeep
├── tests/
│   ├── fixtures/.gitkeep
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_cli.py
│   ├── test_logging.py
│   ├── test_ntnu_loader.py
│   ├── test_terrain_predictor.py
│   └── test_api.py
├── docs/{dataset.md,research_plan.md,lab10_report.md,
│         lab10_validation.json,lab10_ai_prompts.md}
├── scripts/check_repository.py
├── scripts/import_ntnu.py
├── scripts/eda_ntnu.py
├── scripts/render_eda_report.py
├── scripts/prepare_lab10_artifacts.py
├── .github/workflows/ci.yml
├── .sourcecraft/ci.yaml
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
├── LICENSE
├── CONTENT_LICENSE.md
└── README.md
```

`models/terrain_predictor.py` содержит сохранённую схему и предобработку;
`api/` содержит работающие маршруты лабораторной. `sources/ntnu_loader.py`
реализует импорт архива. Часть модулей `pipelines/` и `storage/` пока остаётся
заготовками будущих этапов.
Данные и обученные артефакты исключены из Git; версия оценок и контрольные
суммы артефактов зафиксированы в документации и реестре.

## Будущие этапы

После лабораторной №10 остаются дополнительная независимая когорта,
проверка спорной разметки NTNU и научный сайт. Совместимость ноутбуков с Colab
пока не проверялась.
Подробности: [план](docs/research_plan.md) и [описание датасета](docs/dataset.md).

## Состояние GitHub

Локальный `origin` связан с `https://github.com/debikde/python-programming.git`.
Текущая работа не отправляется на GitHub. Тег `lab1` создаётся только после
отдельной проверки и согласования. Перед будущей публикацией нужно проверить
историю и основную ветку удалённого репозитория, согласовать их с локальной
историей, затем отдельно проверить первый запуск Actions и бейдж.
Локальные коммиты и модели не отправлялись в удалённый репозиторий.
