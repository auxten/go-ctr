import type { GlobalTheme, GlobalThemeOverrides } from 'naive-ui'

const vars: Partial<GlobalTheme['common']> = {
  fontFamily: "'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif !important",
}

const themeOverrides: GlobalThemeOverrides = {
  common: {
    ...vars,
  },
}

export {
  themeOverrides,
}
