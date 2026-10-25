import eslint from '@eslint/js'
import vue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'

export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  ...vue.configs['flat/essential'],
  {
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: { parser: tseslint.parser },
      globals: {
        File: 'readonly', Event: 'readonly', HTMLInputElement: 'readonly',
        URL: 'readonly', Blob: 'readonly', document: 'readonly',
      },
    },
  },
  { rules: { 'vue/multi-word-component-names': 'off' } },
]
