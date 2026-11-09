// @vitest-environment jsdom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App.vue'
import { modelInfo, parseRecords, predictCsv, predictRows, resultsCsv, uploadModel } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('prepared stride input', () => {
  it('accepts one or several complete numeric cycles', () => {
    expect(parseRecords('[1,2]', 2)).toEqual([[1, 2]])
    expect(parseRecords('[[1,2],[3,4]]', 2)).toHaveLength(2)
  })

  it('rejects samples with the wrong schema', () => {
    expect(() => parseRecords('[[1]]', 2)).toThrow('2 чисел')
    expect(() => parseRecords('[]', 2)).toThrow()
    expect(() => parseRecords('[[1,"2"]]', 2)).toThrow()
  })
})

describe('API and result export', () => {
  it('uses the bundled model unless another registered model is selected', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ loaded: true, rows: [], model_name: 'experimental_16_people' }) })
    vi.stubGlobal('fetch', fetchMock)
    await modelInfo()
    expect(fetchMock.mock.calls[0][1].headers).toEqual({})
    await predictRows([[0]], 'experimental_16_people')
    expect(fetchMock.mock.calls[1][1].headers['X-Terrain-Model']).toBe('experimental_16_people')
    await predictCsv(new File(['x\n1'], 'x.csv'), undefined, 'experimental_16_people')
    const csvForm = fetchMock.mock.calls[2][1].body as FormData
    expect(csvForm.has('side')).toBe(false)
    expect(fetchMock.mock.calls[2][1].headers['X-Terrain-Model']).toBe('experimental_16_people')
    const uploaded = await uploadModel(new File(['model'], 'model.joblib'))
    expect(uploaded.model_name).toBe('experimental_16_people')
  })

  it('preserves input columns and escapes CSV text', () => {
    const csv = resultsCsv({ unit: 'step_cycle', input_columns: ['feature', 'note'], rows: [{
      input: { feature: 1, note: 'a,b"c' }, predicted_terrain: 2,
      probabilities: { '2': 0.8 },
    }] })
    expect(csv).toContain('feature,note,row,step,run,samples,true_terrain,predicted_terrain')
    expect(csv).toContain('1,"a,b""c"')
  })
})

describe('application flow', () => {
  it('starts with the working model and removes stale predictions after an error', async () => {
    let failPrediction = false
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith('/model-info')) {
        return { ok: true, json: async () => ({
          loaded: true, model_name: 'working_20_people', feature_names: Array(140).fill('x'),
          training_people: Array(20).fill('person'), trusted_models: ['working_20_people'],
          limitations: [], loaded_models: ['working_20_people'],
        }) }
      }
      if (failPrediction) return { ok: false, json: async () => ({ detail: 'Ошибка модели' }) }
      return { ok: true, json: async () => ({ unit: 'step_cycle', input_columns: ['x'],
        rows: [{ input: { x: 0 }, predicted_terrain: 2, probabilities: { '2': 1 } }],
      }) }
    })
    vi.stubGlobal('fetch', fetchMock)
    const wrapper = mount(App)
    await flushPromises()
    expect(wrapper.text()).toContain('Модель загружена')
    expect(wrapper.text()).toContain('working_20_people')
    expect((wrapper.find('select').element as HTMLSelectElement).value).toBe('')
    await wrapper.find('textarea').setValue(JSON.stringify(Array(140).fill(0)))
    const predictButton = wrapper.findAll('button').find((button) => button.text() === 'Предсказать')!
    await predictButton.trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('Предсказания по циклам')
    failPrediction = true
    await predictButton.trigger('click')
    await flushPromises()
    expect(wrapper.text()).not.toContain('Предсказания по циклам')
    expect(wrapper.find('[role="alert"]').text()).toContain('Ошибка модели')
    wrapper.unmount()
  })
})
