# Assistive Gait Research

[![CI](https://github.com/debikde/python-programming/actions/workflows/ci.yml/badge.svg)](https://github.com/debikde/python-programming/actions/workflows/ci.yml)

Исследовательский проект для анализа двигательной активности пользователей
протезов нижних конечностей по данным носимых инерциальных датчиков (IMU).
Будущая задача ML — классификация условий ходьбы.
Планируемый источник: **IMU data from lower-limb prosthetic users traversing
real-world terrain** (NTNU), [DOI 10.18710/U8RGDL](https://doi.org/10.18710/U8RGDL).
Это исследовательский прототип, а не клиническая система диагностики.

## Текущий этап: ЛР1, базовый и «Базовый+»

Реализованы устанавливаемый пакет `research_graph`, CLI `rg`, конфигурация,
логирование, изолированные тесты, строгая типизация, pre-commit и CI.
Название Python-дистрибутива — `assistive-gait-research`; адрес существующего
GitHub-репозитория — `debikde/python-programming`. Переименование удалённого
репозитория для работы пакета не требуется.

Будущие модули пока содержат только docstrings. Данные не загружены,
модели не обучены, вычислительные эксперименты не проводились.
API, frontend и научный сайт пока не реализованы.
Бейдж отражает удалённый workflow; текущие локальные изменения ещё не опубликованы.

## Окружение и установка

Требуются Git, uv и **Python 3.12** (версия указана в `.python-version`).
Стек выполнения: Pydantic, pydantic-settings, Typer.
Инструменты разработки: Ruff, mypy, pytest, pytest-cov, pre-commit,
pre-commit-hooks. ML- и веб-зависимостей нет.

Установите uv по [официальной инструкции](https://docs.astral.sh/uv/getting-started/installation/)
и убедитесь, что каталог с `uv` доступен в `PATH`:

```powershell
uv --version
uv python install 3.12
```

В текущей локальной папке проекта:

```powershell
uv sync --locked
```

`uv sync` также поддерживается. Для воспроизводимой установки используйте
`--locked`: несовпадение `pyproject.toml` и `uv.lock` станет ошибкой.
Первичная установка требует интернета; сами тесты работают без сети.
После будущей публикации можно будет клонировать репозиторий:

```powershell
git clone https://github.com/debikde/python-programming.git
cd python-programming
uv sync --locked
```

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
`--strict-markers`, branch coverage и порог 70%. Тесты изолируют `RG_*`, `.env`,
кэш настроек и логирование; используют временные каталоги и не требуют
реальных данных, токенов или интернета.

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
│   ├── models/{__init__.py,schemas.py}
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── storage/{__init__.py,artifacts.py}
│   └── api/{__init__.py,main.py,routes.py,inference.py}
├── frontend/README.md
├── website/{README.md,docs/index.md}
├── notebooks/README.md
├── data/{raw,processed}/.gitkeep
├── artifacts/{models,reports}/.gitkeep
├── tests/
│   ├── fixtures/.gitkeep
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_cli.py
│   └── test_logging.py
├── docs/{dataset.md,research_plan.md,lab1_report.md}
├── scripts/check_repository.py
├── .github/workflows/ci.yml
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

`models/` предназначен для схем данных, а обученные модели будут храниться
в `artifacts/models/`. Все файлы в `sources`, `models`, `pipelines`, `storage`
и `api` — заготовки будущих этапов. Данные и артефакты исключены из Git;
сохраняются только маркеры каталогов. `tests/fixtures/` зарезервирован для
небольших тестовых примеров, которые появятся при работе с данными.

## Будущие этапы

Подключение NTNU → EDA → подготовка сигналов и признаков → сравнение минимум
двух моделей → сохранение модели → FastAPI (`/upload-model`, `/predict`,
`/predict-from-csv`) → frontend → научный сайт на MkDocs → итоговый CI в SourceCraft.
Локальные и Colab-ноутбуки появятся на этапе исследований.
Подробности: [план](docs/research_plan.md) и [описание датасета](docs/dataset.md).

## Состояние GitHub

Локальный `origin` связан с `https://github.com/debikde/python-programming.git`.
Текущая работа не отправляется на GitHub. Тег `lab1` создаётся только после
отдельной проверки и согласования. Перед будущей публикацией нужно проверить
историю и основную ветку удалённого репозитория, согласовать их с локальной
историей, затем отдельно проверить первый запуск Actions и бейдж.
Следующая лабораторная не входит в текущий этап.
