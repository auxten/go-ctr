import type { GlobalTheme, GlobalThemeOverrides } from 'naive-ui'

const vars: Partial<GlobalTheme['common']> = {
  fontFamily: "'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif !important",
}

const themeOverrides: GlobalThemeOverrides = {
  common: {
    ...vars,
  },
  Menu: {
    itemColorActiveInverted: '#263445',
    itemColorActiveHoverInverted: '#263445',
    itemColorActiveCollapsedInverted: '#263445',
  },
  Card: {
    // boxShadow: '4px 4px 40px rgb(0 0 0 / 5%)',
    borderRadius: '4px',
  },
}

export {
  themeOverrides,
}
