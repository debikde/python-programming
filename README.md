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

Лабораторный пример с ипотекой адаптирован к циклам шага NTNU. Сервис использует Random Forest без каденса, с 140 признаками. Рабочая модель, обученная на 20 участниках, включена в Python-пакет и загружается при старте API. Исследовательские копии артефактов лежат в `../research-publication-data/artifacts/reports/model_validation/`. Повторно подготовить их можно командой:

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

Откройте `http://127.0.0.1:5173`: рабочая модель уже готова. Выберите PS CSV из папки NTNU и явно укажите сторону протеза либо загрузите CSV с готовыми признаками без выбора стороны. Также доступен ручной ввод признаков. Ответ содержит исходные признаки цикла, класс и вероятности. Скачиваемый CSV сохраняет все столбцы подготовленного файла; для PS CSV он содержит рассчитанные признаки каждого цикла, а не отдельные отсчёты датчика. При наличии меток `Terrain` сервис показывает метрики; ROC-AUC рассчитывается только при наличии всех классов. Загрузка другого зарегистрированного `.joblib` необязательна и меняет модель только для запросов текущей вкладки. После перезапуска API дополнительную модель нужно загрузить снова; рабочая доступна автоматически.

Маршруты: `POST /upload-model`, `POST /predict`, `POST /predict-from-csv`, `GET /model-info`. Примеры запросов, ручная проверка и промпты разработки интерфейса приведены в [отчёте](docs/lab10_report.md); интерактивная схема доступна по `http://127.0.0.1:8000/docs`. Список доверенных контрольных сумм — в `src/research_graph/api/trusted_models.json`. Переобученный файл нужно проверять перед добавлением в этот список.

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

Проверки запускаются в GitHub Actions через `.github/workflows/ci.yml`. `pytest` измеряет покрытие ветвей пакета `research_graph` через `pytest-cov` и требует не меньше 70%. Обычные push и pull request используют GitHub-hosted runner. CI проверяет Vue и сохраняет wheel Python-пакета, собранный frontend и отчёт о покрытии как артефакты. Git-репозиторий связан с [debikde/python-programming](https://github.com/debikde/python-programming), но локальные изменения не отправляются автоматически.

### Ручная демонстрация на Windows runner

Репозиторий публичный. Задание `Windows runner demonstration` в текущем workflow запускается только вручную; обычные проверки pull request остаются на GitHub-hosted runner. Для self-hosted runner нужна отдельная Windows-машина или чистая виртуальная машина с доступом к GitHub. Не используйте для этого основной компьютер с личными файлами и токенами: другие workflow публичного репозитория тоже могут направить задания на зарегистрированный runner.

После появления workflow в основной ветке GitHub:

1. На выделенной Windows-машине получите ветку `lab10`: `git clone --branch lab10 https://github.com/debikde/python-programming.git`, затем перейдите в `python-programming`. Выполните `powershell -ExecutionPolicy Bypass -File scripts/prepare_windows_runner.ps1`. Скрипт скачает версию runner `2.338.2` в `C:\actions-runner`, сверит SHA-256 с официальным релизом и распакует файлы. Токен в скрипте не хранится.
2. Откройте **Settings → Actions → Runners → New self-hosted runner** в репозитории и получите новый регистрационный токен. Он действует один час. Не используйте токен, уже вставленный в переписку, и не сохраняйте новый токен в репозитории. В PowerShell на той же машине выполните:

   ```powershell
   cd C:\actions-runner
   $runnerToken = Read-Host 'Новый временный токен GitHub runner'
   .\config.cmd --url https://github.com/debikde/python-programming --token $runnerToken --labels gait-lab --name gait-lab-windows --unattended
   Remove-Variable runnerToken
   .\run.cmd
   ```

3. Когда runner показывает `Listening for Jobs`, откройте **Actions → CI → Run workflow** и выберите ветку `lab10`. Задание проверит Python, покрытие, frontend и соберёт wheel и статические файлы Vue. Их можно скачать из артефакта `lab10-windows-demo` вместе с `coverage.xml`.

Сервер и автоматическая публикация не нужны для этой демонстрации. Исходные CSV остаются вне Git. Проверки используют небольшие тестовые данные и включённый в пакет обученный артефакт; на runner не нужно копировать исследовательскую папку. Показ работы с настоящей моделью выполняется локально по разделу «API и интерфейс». Команда `run.cmd` должна оставаться запущенной, пока нужны задания; после демонстрации runner можно остановить.

Python-тесты находятся в двух файлах: `tests/test_api.py` проверяет загрузку модели и предсказания, `tests/test_research.py` — импорт NTNU, обработку сигналов и CLI. Тесты интерфейса находятся в `frontend/src/api.test.ts`. Исследовательские результаты сверяются по сохранённым таблицам и выполненным ноутбукам.
