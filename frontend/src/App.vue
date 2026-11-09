<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  TERRAIN_CODES, modelInfo, parseRecords, predictCsv, predictRows, resultsCsv, uploadModel,
  type ModelInfo, type PredictionResponse,
} from './api'

const info = ref<ModelInfo | null>(null)
const result = ref<PredictionResponse | null>(null)
const busy = ref(false)
const error = ref('')
const csvFile = ref<File | null>(null)
const side = ref('')
const selectedModelName = ref('')
const manualInput = ref('')

function percent(value: number | null | undefined): string {
  return value == null ? 'не определён' : `${(value * 100).toFixed(1)}%`
}

async function refresh(): Promise<void> {
  try {
    info.value = await modelInfo(selectedModelName.value || undefined)
  } catch (caught) {
    error.value = `API недоступен: ${String(caught)}`
  }
}

async function run(action: () => Promise<void>): Promise<void> {
  busy.value = true
  error.value = ''
  result.value = null
  try {
    await action()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  } finally {
    busy.value = false
  }
}

function selectedFile(event: Event): File | null {
  return (event.target as HTMLInputElement).files?.[0] ?? null
}

function onModelChange(event: Event): void {
  const file = selectedFile(event)
  if (!file) return
  void run(async () => {
    const uploaded = await uploadModel(file)
    selectedModelName.value = uploaded.model_name
    await refresh()
  })
}

function onCsvChange(event: Event): void {
  csvFile.value = selectedFile(event)
  result.value = null
}

function useDefaultModel(): void {
  selectedModelName.value = ''
  result.value = null
  void refresh()
}

function submitCsv(): void {
  if (!csvFile.value) {
    error.value = 'Сначала выберите CSV-файл.'
    return
  }
  void run(async () => { result.value = await predictCsv(csvFile.value!, side.value, selectedModelName.value) })
}

function submitManual(): void {
  void run(async () => {
    const records = parseRecords(manualInput.value, info.value?.feature_names?.length ?? 140)
    result.value = await predictRows(records, selectedModelName.value)
  })
}

function saveFile(name: string, content: string): void {
  const href = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }))
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = name
  anchor.click()
  URL.revokeObjectURL(href)
}

function downloadTemplate(): void {
  if (info.value?.feature_names) {
    saveFile('terrain_features_template.csv', `${info.value.feature_names.join(',')}\n`)
  }
}

function downloadResults(): void {
  if (!result.value) return
  saveFile('terrain_predictions.csv', resultsCsv(result.value))
}

onMounted(() => { void refresh() })
</script>

<template>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Assistive Gait Research / лабораторная №10</div>
      <h1>Классификация поверхности по IMU</h1>
      <p>Одна запись результата — один цикл шага. Сервис принимает PS-данные NTNU
        или уже подготовленные 140 признаков и возвращает числовой код Terrain.</p>
      <div class="hero-tags"><span>20 участников NTNU</span><span>8 кодов Terrain</span><span>Random Forest</span></div>
    </header>

    <div v-if="error" class="error" role="alert">{{ error }}</div>

    <section class="card" aria-labelledby="model-title">
      <div class="section-head">
        <div><span class="step">01 / Модель</span><h2 id="model-title">Модель классификации</h2></div>
        <span class="status" :class="{ ready: info?.loaded }">
          {{ info?.loaded ? 'Модель загружена' : 'Модель не загружена' }}
        </span>
      </div>
      <p>Рабочая модель обучена на 20 участниках и загружается при запуске API.
        Другой зарегистрированный артефакт можно выбрать дополнительно.</p>
      <label class="file-picker">
        <span>Загрузить другую модель .joblib (необязательно)</span>
        <input type="file" accept=".joblib" :disabled="busy" @change="onModelChange">
      </label>
      <button v-if="selectedModelName" class="secondary" :disabled="busy" @click="useDefaultModel">Вернуться к рабочей модели</button>
      <p v-if="info?.loaded" class="hint">{{ info.model_name }} ·
        {{ info.feature_names?.length }} признаков · {{ info.training_people?.length }} участников в обучении</p>
      <p v-else class="hint">Доступные ID: {{ info?.trusted_models.join(', ') || 'загрузка списка…' }}</p>
    </section>

    <section class="card" aria-labelledby="input-title">
      <div class="section-head"><div><span class="step">02 / Данные</span><h2 id="input-title">Получить предсказания</h2></div></div>
      <div class="input-grid">
        <div class="input-panel">
          <h3>CSV-файл</h3>
          <p>PS IMU: отсчёты датчика, столбец <code>Steps</code> и сторона протеза.
            Подготовленный CSV: одна строка на цикл и имена 140 признаков.</p>
          <input type="file" accept=".csv,text/csv" :disabled="busy || !info?.loaded" @change="onCsvChange">
          <label class="field">Сторона протеза для PS CSV
            <select v-model="side"><option value="">Выберите для PS CSV</option><option value="left">Левая</option><option value="right">Правая</option></select>
          </label>
          <p class="hint">Для CSV с готовыми признаками сторона не нужна. Для PS CSV её нужно выбрать явно.</p>
          <button :disabled="busy || !info?.loaded || !csvFile" @click="submitCsv">Обработать CSV</button>
          <button class="secondary" :disabled="!info?.loaded" @click="downloadTemplate">Шаблон признаков</button>
        </div>
        <div class="input-panel">
          <h3>Подготовленные циклы вручную</h3>
          <p>Вставьте JSON-массив из 140 чисел для одного цикла или массив таких строк
            для нескольких циклов. Порядок признаков указан ниже.</p>
          <textarea v-model="manualInput" rows="6" placeholder="[[0.1, 0.2, … 140 чисел]]" :disabled="busy || !info?.loaded" />
          <button :disabled="busy || !info?.loaded" @click="submitManual">Предсказать</button>
        </div>
      </div>
      <details v-if="info?.loaded" class="feature-list">
        <summary>Порядок 140 признаков для ручного ввода</summary>
        <p>{{ info.feature_names?.join(' · ') }}</p>
      </details>
    </section>

    <section v-if="result" class="card" aria-labelledby="result-title">
      <div class="section-head">
        <div><span class="step">03 / Результат</span><h2 id="result-title">Предсказания по циклам</h2></div>
        <button class="secondary" @click="downloadResults">Скачать все строки</button>
      </div>
      <p>{{ result.rows.length }} циклов · показаны первые {{ Math.min(100, result.rows.length) }}.
        Вероятности относятся к числовым кодам Terrain.</p>
      <div v-if="result.metrics" class="metrics">
        <div><small>Accuracy</small><strong>{{ percent(result.metrics.accuracy) }}</strong></div>
        <div><small>Macro-F1</small><strong>{{ percent(result.metrics.macro_f1) }}</strong></div>
        <div><small>Balanced accuracy</small><strong>{{ percent(result.metrics.balanced_accuracy) }}</strong></div>
        <div><small>ROC-AUC OvR</small><strong>{{ percent(result.metrics.roc_auc_ovr) }}</strong></div>
      </div>
      <p v-if="result.metrics" class="hint">Метрики рассчитаны по {{ result.labelled_cycles }} циклам с допустимой истинной меткой.
        ROC-AUC определяется только при наличии всех восьми классов.</p>
      <div class="table-scroll"><table><thead><tr>
        <th>Цикл</th><th>Шаг</th><th>Истина</th><th>Прогноз</th>
        <th v-for="code in TERRAIN_CODES" :key="code">P({{ code }})</th>
      </tr></thead><tbody><tr v-for="(row, index) in result.rows.slice(0, 100)" :key="index">
        <td>{{ row.row ?? row.run ?? index }}</td><td>{{ row.step ?? '—' }}</td>
        <td>{{ row.true_terrain ?? '—' }}</td><td class="prediction">{{ row.predicted_terrain }}</td>
        <td v-for="code in TERRAIN_CODES" :key="code">{{ percent(row.probabilities[String(code)]) }}</td>
      </tr></tbody></table></div>
    </section>

    <section class="card" aria-labelledby="quality-title">
      <div class="section-head"><div><span class="step">04 / Контекст</span><h2 id="quality-title">Качество и ограничения</h2></div></div>
      <p>Внешние LSO-оценки ниже относятся к заранее заданным конфигурациям на тех же 20 участниках.
        Выбор рабочей конфигурации выполнен отдельно по 16 участникам, без подбора на отложенной группе.</p>
      <div v-if="info?.validation" class="table-scroll"><table><thead><tr>
        <th>Модель</th><th>ROC-AUC OvR</th><th>Accuracy</th><th>Macro-F1</th><th>Balanced accuracy</th>
      </tr></thead><tbody><tr v-for="item in info.validation.five_fold_lso_fixed_configs" :key="item.algorithm">
        <td>{{ item.algorithm }}</td><td>{{ percent(item.roc_auc_ovr) }}</td>
        <td>{{ percent(item.accuracy) }}</td><td>{{ percent(item.macro_f1) }}</td>
        <td>{{ percent(item.balanced_accuracy) }}</td>
      </tr></tbody></table></div>
      <p v-if="info?.validation" class="highlight">Отложенные 4 участника: accuracy
        {{ percent(info.validation.experimental_model.holdout_metrics.accuracy) }}, macro-F1
        {{ percent(info.validation.experimental_model.holdout_metrics.macro_f1) }}, ROC-AUC OvR
        {{ percent(info.validation.experimental_model.holdout_metrics.roc_auc_ovr) }}.
        Это оценка экспериментального артефакта, обученного на 16 людях.
        Рабочая копия обучена на всех 20 и отдельного независимого теста не имеет.</p>
      <ul class="limits"><li v-for="item in info?.limitations ?? []" :key="item">{{ item }}</li>
        <li>Магнитометр заметно влияет на качество. В отложенной группе коды 6 и 7 не были распознаны.</li>
        <li>Разметка NTNU содержит неоднозначности; названия для кодов 8 и 9 не утверждены.</li>
      </ul>
    </section>
    <footer>Исследовательский прототип · не использовать для клинических решений</footer>
  </main>
</template>
