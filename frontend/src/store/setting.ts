import { defineStore } from 'pinia'
import { defaultSettings } from '@/settings'

const { showSettings, tagsView, fixedHeader, sidebarLogo } = defaultSettings

export const useSettingStore = defineStore('setting', {
  state: () => ({
    theme: 'light',
    showSettings: showSettings,
    tagsView: tagsView,
    fixedHeader: fixedHeader,
    sidebarLogo: sidebarLogo,
  }),
  actions: {
    changeSetting(key: string, value: any) {
      this[key] = value
    },
  },
})
