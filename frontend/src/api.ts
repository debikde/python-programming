export const TERRAIN_CODES = [1, 2, 4, 5, 6, 7, 8, 9] as const

export type Metrics = {
  accuracy: number
  macro_f1: number
  balanced_accuracy: number
  roc_auc_ovr: number | null
}

export type PredictionRow = {
  input: Record<string, string | number | null>
  row?: number
  step?: number
  run?: number
  samples?: number
  true_terrain?: number | null
  predicted_terrain: number
  probabilities: Record<string, number>
}

export type Validation = {
  experimental_model: { holdout_metrics: Metrics; heldout_people: string[] }
  five_fold_lso_fixed_configs: Array<{ algorithm: string } & Metrics>
}

export type ModelInfo = {
  loaded: boolean
  model_name?: string
  default_model_name?: string
  loaded_models: string[]
  feature_names?: string[]
  classes?: number[]
  trusted_models: string[]
  limitations: string[]
  validation?: Validation
  training_people?: string[]
}

export type PredictionResponse = {
  unit: string
  input_columns: string[]
  input_kind?: string
  rows: PredictionRow[]
  metrics?: Metrics | null
  labelled_cycles?: number
}

const base = import.meta.env.VITE_API_BASE_URL || '/api'

async function read<T>(response: Response): Promise<T> {
  const body = await response.json()
  if (!response.ok) {
    const detail = body.detail
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return body as T
}

function modelHeader(modelName?: string): Record<string, string> {
  return modelName ? { 'X-Terrain-Model': modelName } : {}
}

export async function modelInfo(modelName?: string): Promise<ModelInfo> {
  return read<ModelInfo>(await fetch(`${base}/model-info`, { headers: modelHeader(modelName) }))
}

export async function uploadModel(file: File): Promise<{ model_name: string }> {
  const form = new FormData()
  form.append('file', file)
  return read(await fetch(`${base}/upload-model`, { method: 'POST', body: form }))
}

export async function predictRows(records: number[][], modelName?: string): Promise<PredictionResponse> {
  return read<PredictionResponse>(await fetch(`${base}/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...modelHeader(modelName) },
    body: JSON.stringify({ records }),
  }))
}

export async function predictCsv(file: File, side?: string, modelName?: string): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)
  if (side) form.append('side', side)
  return read<PredictionResponse>(await fetch(`${base}/predict-from-csv`, {
    method: 'POST', body: form, headers: modelHeader(modelName),
  }))
}

export function resultsCsv(result: PredictionResponse): string {
  const headings = [...result.input_columns, 'row', 'step', 'run', 'samples',
    'true_terrain', 'predicted_terrain',
    ...TERRAIN_CODES.map((code) => `probability_${code}`)]
  const cell = (value: string | number | null | undefined): string => {
    const text = value == null ? '' : String(value)
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  }
  const lines = result.rows.map((row) => [
    ...result.input_columns.map((name) => row.input[name]),
    row.row, row.step, row.run, row.samples, row.true_terrain, row.predicted_terrain,
    ...TERRAIN_CODES.map((code) => row.probabilities[String(code)]),
  ].map(cell).join(','))
  return [headings.map(cell).join(','), ...lines].join('\n')
}

export function parseRecords(text: string, featureCount: number): number[][] {
  const parsed: unknown = JSON.parse(text)
  const records = Array.isArray(parsed) && parsed.every((value) => typeof value === 'number')
    ? [parsed] : parsed
  if (!Array.isArray(records) || records.length === 0 || records.length > 1000 ||
      !records.every((row) => Array.isArray(row) && row.length === featureCount &&
        row.every((value) => typeof value === 'number' && Number.isFinite(value)))) {
    throw new Error(`Нужен массив из 1–1000 циклов по ${featureCount} чисел`)
  }
  return records as number[][]
}
