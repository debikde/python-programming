export const TERRAIN_CODES = [1, 2, 4, 5, 6, 7, 8, 9] as const

export type Metrics = {
  accuracy: number
  macro_f1: number
  balanced_accuracy: number
  roc_auc_ovr: number | null
}

export type PredictionRow = {
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
  feature_names?: string[]
  classes?: number[]
  trusted_models: string[]
  limitations: string[]
  validation?: Validation
  training_people?: string[]
}

export type PredictionResponse = {
  unit: string
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

export async function modelInfo(): Promise<ModelInfo> {
  return read<ModelInfo>(await fetch(`${base}/model-info`))
}

export async function uploadModel(file: File): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  await read(await fetch(`${base}/upload-model`, { method: 'POST', body: form }))
}

export async function predictRows(records: number[][]): Promise<PredictionResponse> {
  return read<PredictionResponse>(await fetch(`${base}/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records }),
  }))
}

export async function predictCsv(file: File, side: string): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('side', side)
  return read<PredictionResponse>(await fetch(`${base}/predict-from-csv`, {
    method: 'POST', body: form,
  }))
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
