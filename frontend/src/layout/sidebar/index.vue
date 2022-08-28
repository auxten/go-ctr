<template>
  <n-layout-sider
    inverted
    collapse-mode="width"
    :width="210"
    :collapsed-width="64"
    :collapsed="isCollapse"
  >
    <div :class="{'has-logo': showLogo}">
      <logo v-if="showLogo" :collapse="isCollapse" />
      <n-scrollbar wrap-class="scrollbar-wrapper">
        <n-menu
          :value="activeMenu"
          inverted
          :collapsed-width="64"
          :options="menuOptions"
        />
      </n-scrollbar>
    </div>
  </n-layout-sider>
</template>

<script lang="ts" setup>
import { h, computed } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { NIcon } from 'naive-ui'
import { IconDashboard } from '@/icons'
import { PeopleOutline, PricetagOutline } from '@vicons/ionicons5'
import { useSettingStore } from '@/store/setting'
import { useAppStore } from '@/store/app'
import { asyncRoutes } from '@/routes'
import logo from './logo.vue'

const route = useRoute()
const router = useRouter()
const settingStore = useSettingStore()
const appStore = useAppStore()

const showLogo = computed(() => {
  return settingStore.sidebarLogo
})

const isCollapse = computed(() => {
  return !appStore.sidebar.opened
})

const activeMenu = computed(() => {
  const { meta, path } = route
  // if set path, the sidebar will highlight the path you set
  if (meta.activeMenu) {
    return meta.activeMenu
  }
  return path
})

const renderIcon = (icon: any) => {
  return () => h(NIcon, null, { default: () => h(icon) })
}

const getIcon = (name: string) => {
  if (name === 'Overview') {
    return renderIcon(IconDashboard)
  } else if (name === 'Users') {
    return renderIcon(PeopleOutline)
  } else if (name === 'Items') {
    return renderIcon(PricetagOutline)
  }
}

const menuOptions = asyncRoutes.map(m => {
  return {
    key: router.resolve({ name: m.name }).path,
    label: m.name,
    icon: getIcon(m.name as string),
    label: () =>
      h(RouterLink, {
        to: {
          name: m.name,
        },
      },
      { default: () => m.name },
      ),
  }
})
</script>
