import { describe, expect, it } from 'vitest'
import { parseRecords } from './api'

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
