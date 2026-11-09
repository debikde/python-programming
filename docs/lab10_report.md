# Лабораторная №10: классификация Terrain по циклам шага

Предмет лабораторной адаптирован к NTNU IMU Dataset. Сервис получает запись
PS-датчика или уже рассчитанные признаки одного цикла шага и возвращает один
из восьми числовых кодов Terrain и вероятности классов. Пример с ипотекой из
учебного задания служит только образцом структуры сервиса.

## Сравнение моделей

В [выполненном ноутбуке](../notebooks/04_model_validation.ipynb) все три
алгоритма проверены на одних и тех же 9781 циклах и пяти LSO-группах людей.
В этой таблице у всех моделей 150 исходных признаков и `StandardScaler`,
обученный только на тренировочной части каждого фолда. Параметры внешних
сравнений фиксированы до просмотра соответствующих тестовых фолдов.

| Модель | ROC-AUC OvR | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0,798 | 0,502 | 0,464 | 0,472 |
| Logistic Regression | 0,786 | 0,439 | 0,428 | 0,467 |
| Extra Trees | 0,786 | 0,466 | 0,396 | 0,370 |

Здесь приведены средние пяти фолдов, а ROC-AUC — многоклассовый OvR macro
по вероятностям, когда в тестовом фолде присутствовали все классы. Random
Forest имеет наибольший ROC-AUC из трёх сопоставимых систем и одновременно
наибольший macro-F1. Это поддерживает выбор алгоритма по требуемому в
лабораторной ROC-AUC. Однако **внешние LSO-результаты не использовались для
подбора**: конкретные `max_features=15` и набор без каденса выбраны на
четырёх внутренних групповых проверках 16 участников. Поэтому таблица выше
не является внешней оценкой точно этой итоговой конфигурации.

## Отложенная группа и рабочая копия

Экспериментальный артефакт обучен на 16 людях. На заранее отложенных P1,
P5, P15 и P19 он получил accuracy **0,414**, macro-F1 **0,299**,
balanced accuracy **0,309**, ROC-AUC OvR **0,655**. Показатели и результаты
по каждому человеку зафиксированы в [JSON](lab10_validation.json), подробные
предсказания — в локальном `holdout_predictions.csv`. Эта группа уже
встречалась в ранних исследованиях проекта; она не является полностью новой
внешней когортой. Ни параметры, ни каналы по её метрикам не менялись.

После фиксации оценки отдельно создан `terrain_predictor_final.joblib`:
та же конфигурация обучена на всех 20 участниках. Этот рабочий артефакт
удобен для демонстрации, но **не имеет собственного независимого теста**.
Метрики экспериментального файла нельзя приписывать этой новой подгонке.
Оба файла содержат `TerrainPredictor`, полный sklearn `Pipeline`, схему 140
признаков, числовые классы и версии библиотек. API принимает только точные
копии файлов с SHA-256, зарегистрированными в репозитории.

Рабочая модель входит в пакет и загружается при старте API. Загружать файл
перед первым предсказанием не требуется. `POST /upload-model` позволяет
выбрать другой зарегистрированный артефакт. Его имя передаётся в заголовке
`X-Terrain-Model`; запросы без заголовка продолжают использовать рабочую
модель. Для исходного PS CSV нужно явно передать сторону протеза. В ответе
сохраняются исходные столбцы подготовленного CSV или рассчитанные признаки
циклов исходного IMU CSV.

## Ограничения

- Внешнее качество на новых людях низкое и неодинаковое: accuracy отдельных
  участников отложенной группы находится между 0,217 и 0,657.
- Recall кодов 6 и 7 на этой группе равен **0**. Поэтому даже умеренная
  общая accuracy не означает надёжное распознавание всех условий.
- Удаление магнитометра у фиксированного Random Forest снизило средний
  macro-F1 с 0,464 до 0,415. Устройство без него потребует отдельной проверки.
- В NTNU остаются вопросы к кодам Terrain 8/9, значениям 0/99, отдельным
  файлам и масштабу P16/P19. Для кодов 8/9 сервис не придумывает названия.
- Частота 100 Гц и сторона протеза передаются по исходному протоколу.
  Иной датчик или популяция требуют новой валидации.

Сервис остаётся исследовательским прототипом, не инструментом клинического
решения. Следующий научный сайт сможет использовать этот отчёт без изменения
сохранённых ноутбуков и метрик.

## Ручная проверка API и интерфейса

Файлы в `examples/lab10/` искусственные. Они проверяют приём данных,
предобработку и ответы сервиса; метрики на них **не показывают качество модели
на людях**. Исходные данные NTNU в примеры не копировались. Файлы можно
пересоздать из корня проекта командой
`.\.venv\Scripts\python.exe scripts/prepare_manual_lab10_files.py`.

### Запуск

Нужны Python 3.12, окружение `.venv`, Node.js и npm. Если `.venv` ещё нет,
выполните `uv sync --locked --all-groups` из корня проекта. На этом
компьютере `uv.exe` находится в
`C:\Users\Zina\AppData\Roaming\Python\Python312\Scripts\`.

Первый терминал PowerShell — backend:

```powershell
cd C:\Users\Zina\research-publication
.\.venv\Scripts\python.exe -m uvicorn research_graph.api.main:app --host 127.0.0.1 --port 8000
```

Второй терминал — frontend:

```powershell
cd C:\Users\Zina\research-publication\frontend
npm ci
npm run dev
```

Оставьте оба терминала открытыми. API доступен по
`http://127.0.0.1:8000/docs`, интерфейс — по `http://127.0.0.1:5173`.
Остановить процессы можно через `Ctrl+C` в соответствующих терминалах.

### Проверка всех маршрутов API

Откройте третий PowerShell. Все следующие команды запускаются из корня
проекта. `curl.exe` сохраняет полные ответы во временной папке и печатает
HTTP-код; эти файлы не попадают в репозиторий.

```powershell
cd C:\Users\Zina\research-publication
$api = 'http://127.0.0.1:8000'
$out = Join-Path $env:TEMP 'gait-lab10-manual'
New-Item -ItemType Directory -Force -Path $out | Out-Null
```

**`GET /health` и `GET /model-info`.**

```powershell
Invoke-RestMethod "$api/health"
$info = Invoke-RestMethod "$api/model-info"
$info | Select-Object loaded, model_name, default_model_name
$info.feature_names.Count
$info.classes
$info.training_people.Count
```

Ожидаются `status: ok`, `loaded: True`, `working_20_people`, 140 признаков,
коды `1, 2, 4, 5, 6, 7, 8, 9` и 20 участников. Модель работает до загрузки
какого-либо файла пользователем.

**`POST /predict`: две подготовленные записи.**

```powershell
curl.exe -sS -o "$out\predict.json" -w "%{http_code}`n" -H "Content-Type: application/json" --data-binary "@examples/lab10/records.json" "$api/predict"
$prediction = Get-Content "$out\predict.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$prediction.unit
$prediction.rows.Count
$prediction.rows[0].predicted_terrain
$prediction.rows[0].probabilities
```

Ожидаются HTTP `200`, `step_cycle` и 2 строки. В каждой строке есть 140
входных значений, один числовой код и восемь вероятностей, сумма которых
близка к 1. Конкретный код предсказания не является критерием проверки.

**`POST /predict-from-csv`: подготовленный CSV с метками.** Сторона
протеза для этого файла не нужна.

```powershell
curl.exe -sS -o "$out\prepared.json" -w "%{http_code}`n" -F "file=@examples/lab10/prepared_labeled.csv" "$api/predict-from-csv"
$prepared = Get-Content "$out\prepared.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$prepared.input_kind
$prepared.rows.Count
$prepared.labelled_cycles
$prepared.metrics
$prepared.rows[0].input.note
```

Ожидаются HTTP `200`, `prepared_cycles`, 8 строк, 8 размеченных циклов,
accuracy, macro-F1, balanced accuracy и ROC-AUC OvR. Последний определён,
поскольку в искусственном примере есть все восемь кодов Terrain. Исходное
значение `note` первой строки — `пример, 1`.

**Тот же маршрут без истинных меток.**

```powershell
curl.exe -sS -o "$out\unlabeled.json" -w "%{http_code}`n" -F "file=@examples/lab10/prepared_unlabeled.csv" "$api/predict-from-csv"
$unlabeled = Get-Content "$out\unlabeled.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$unlabeled.rows.Count
$unlabeled.labelled_cycles
$unlabeled.metrics
```

Ожидаются HTTP `200`, 2 строки, `labelled_cycles: 0` и `metrics: null`.

**Исходные отсчёты PS IMU.** Файл содержит 280 отсчётов в 8 циклах. Для
этой искусственной записи выберите `left`.

```powershell
curl.exe -sS -o "$out\imu.json" -w "%{http_code}`n" -F "side=left" -F "file=@examples/lab10/imu_labeled.csv" "$api/predict-from-csv"
$imu = Get-Content "$out\imu.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$imu.input_kind
$imu.rows.Count
$imu.labelled_cycles
$imu.metrics
$imu.rows[0] | Select-Object step, run, samples, predicted_terrain
```

Ожидаются HTTP `200`, `imu_cycles`, 8 строк и 8 размеченных циклов. Одна
строка ответа соответствует циклу шага, а не одному отсчёту датчика.
Входные значения в ответе — 140 рассчитанных признаков цикла.

**Ошибки валидации.** Каждая команда должна напечатать указанный HTTP-код.
Объяснение ошибки сохранится в JSON-файле во временной папке.

```powershell
# Неверное число признаков: 422
curl.exe -sS -o "$out\bad-records.json" -w "%{http_code}`n" -H "Content-Type: application/json" --data-binary "@examples/lab10/invalid_records.json" "$api/predict"
# PS CSV без стороны: 422
curl.exe -sS -o "$out\no-side.json" -w "%{http_code}`n" -F "file=@examples/lab10/imu_labeled.csv" "$api/predict-from-csv"
# CSV без нужных столбцов: 422
curl.exe -sS -o "$out\bad-csv.json" -w "%{http_code}`n" -F "file=@examples/lab10/invalid_columns.csv" "$api/predict-from-csv"
# Неизвестная выбранная модель: 409
curl.exe -sS -o "$out\unknown-model.json" -w "%{http_code}`n" -H "Content-Type: application/json" -H "X-Terrain-Model: unknown" --data-binary "@examples/lab10/records.json" "$api/predict"
# Недоверенный .joblib: 400
curl.exe -sS -o "$out\untrusted.json" -w "%{http_code}`n" -F "file=@examples/lab10/untrusted_model.joblib" "$api/upload-model"
```

**`POST /upload-model`: зарегистрированный дополнительный артефакт.**

```powershell
curl.exe -sS -o "$out\upload.json" -w "%{http_code}`n" -F "file=@src/research_graph/api/assets/terrain_predictor.joblib" "$api/upload-model"
Get-Content "$out\upload.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$other = Invoke-RestMethod "$api/model-info" -Headers @{'X-Terrain-Model'='experimental_16_people'}
$other.model_name
$other.training_people.Count
(Invoke-RestMethod "$api/model-info").model_name
curl.exe -sS -o "$out\experimental.json" -w "%{http_code}`n" -H "Content-Type: application/json" -H "X-Terrain-Model: experimental_16_people" --data-binary "@examples/lab10/records.json" "$api/predict"
```

Загрузка и предсказание возвращают HTTP `200`. Дополнительная модель имеет
имя `experimental_16_people` и 16 обучающих участников. Без заголовка
`X-Terrain-Model` сервис по-прежнему выбирает `working_20_people`.

### Проверка всего интерфейса

1. Откройте `http://127.0.0.1:5173`. Должны появиться статус «Модель
   загружена», имя `working_20_people`, 140 признаков и 20 участников.
   Загрузка файла модели перед первым предсказанием не требуется. В нижнем
   разделе видны таблица сравнения моделей и ограничения.
2. Нажмите «Шаблон признаков». Скачанный CSV должен содержать 140 заголовков.
   Готовый пример с такими признаками — `examples/lab10/prepared_labeled.csv`.
3. Выберите `examples/lab10/prepared_labeled.csv`, оставьте сторону пустой и
   нажмите «Обработать CSV». Должны появиться 8 строк, истинные и
   предсказанные коды, вероятности и четыре метрики. Нажмите «Скачать все
   строки». В скачанном файле должны сохраниться входные признаки, `Terrain`,
   `note` и предсказания. Текст `пример, 1` должен остаться в одном столбце.
4. Выберите `examples/lab10/prepared_unlabeled.csv`. Ожидаются 2 строки
   предсказаний без блока метрик.
5. Выберите `examples/lab10/imu_labeled.csv` и оставьте сторону пустой.
   Ожидается ошибка с просьбой указать `left` или `right`; прежние
   предсказания исчезают. Выберите «Левая» и повторите запрос: должны
   появиться 8 циклов с метриками. В скачанном CSV находятся признаки циклов,
   а не все 280 отсчётов. Можно также выбрать «Правая» для проверки второй
   ветви предобработки; значения метрик на искусственном файле не оценивают
   правильность выбора стороны.
6. Откройте `examples/lab10/manual_records.json` в текстовом редакторе.
   Скопируйте весь JSON-массив в поле ручного ввода и нажмите «Предсказать»:
   должны появиться 2 результата. Затем вставьте `[[1, 2]]` и повторите:
   появится ошибка о числе признаков, а прежние результаты исчезнут.
7. В разделе модели загрузите
   `src/research_graph/api/assets/terrain_predictor.joblib`. Должны
   отображаться `experimental_16_people` и 16 участников. Повторите
   предсказание по `prepared_labeled.csv`. Нажмите «Вернуться к рабочей
   модели»: снова появятся `working_20_people` и 20 участников.
8. Загрузите `examples/lab10/untrusted_model.joblib`. Ожидается сообщение
   об ошибке контрольной суммы. Такой файл не активируется.
9. Перезапустите backend и обновите страницу. Рабочая модель снова доступна
   сразу. Экспериментальный артефакт, если он нужен, придётся загрузить
   повторно.

Если на первом шаге показано «API недоступен», проверьте первый терминал и
`http://127.0.0.1:8000/health`. Адрес frontend берите из вывода Vite во
втором терминале. В режиме разработки Vite пересылает запросы `/api` на
локальный backend.

## Использование ИИ при разработке интерфейса

Для программной части использовался Codex. Основные пользовательские запросы,
по которым создавался и исправлялся интерфейс, приведены здесь вместо
отдельного файла:

> Создай простой frontend на Vue 3 + TypeScript. Нужны загрузка модели, ввод
> или загрузка данных, таблица предсказаний, вероятности классов и информация
> о качестве модели. Показывай ограничения модели. Для кодов Terrain 8/9 не
> используй неподтверждённые названия.

> Доработай указанные недочеты. Плюс мне не нравится, что пользователю
> необходима своя модель в приложении. Я хочу, чтобы уже обученная модель
> запускалась по умолчанию сама в системе, а добавить другую можно уже как
> опция.

> Окей, далее пойдем прорабатывать бек и фронтенд. Перепроверь реализации на
> то, как они работают, соответствуют ли требованиям лабы. Дай отчет об их
> работе и способе запуска. Какой на данный момент условный юзер флоу.

> Верни только `lab10_report.md`. И дай мне инструкцию подробную с
> подготовленными файлами для полного ручного тестирования: тестирование всех
> API, тестирование всего фронта, загрузка модели, работа с моделью и
> получение результатов.

Задание по лабораторной применялось к данным NTNU, а не к примеру с
ипотекой. Код и ответы проверялись тестами и ручными запросами. Временные
токены для регистрации GitHub runner в отчёт не включены.
