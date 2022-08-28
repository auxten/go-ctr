<template>
  <n-layout has-sider :class="classObj" class="app-wrapper">
    <sidebar />
    <n-layout :class="{hasTagsView: settingStore.tagsView}">
      <n-layout-header :class="{'fixed-header': settingStore.fixedHeader}">
        <navbar />
        <tags-view v-if="settingStore.tagsView" />
      </n-layout-header>
      <n-layout-content
        content-style="padding: 20px 24px;"
        class="app-content"
        style="margin-top: 4px;"
      >
        <app-main />
        <right-panel v-if="settingStore.showSettings">
          <settings />
        </right-panel>
      </n-layout-content>
    </n-layout>
  </n-layout>
</template>

<script lang="ts" setup>
import { computed } from 'vue'
import { useSettingStore } from '@/store/setting'
import { useAppStore } from '@/store/app'
import Sidebar from './sidebar/index.vue'
import Navbar from './navbar.vue'
import TagsView from './tags-view/index.vue'
import AppMain from './app-main.vue'
import RightPanel from './right-panel.vue'
import Settings from './settings.vue'

const appStore = useAppStore()
const settingStore = useSettingStore()

const classObj = computed(() => {
  return {
    hideSidebar: !appStore.sidebar.opened,
    openSidebar: appStore.sidebar.opened,
    withoutAnimation: appStore.sidebar.withoutAnimation,
  }
})
</script>

<style lang="scss" scoped>
.app-wrapper {
  position: relative;
  height: 100%;
  width: 100%;

  &::after {
    content: "";
    display: table;
    clear: both;
  }
}

.fixed-header {
  position: fixed;
  top: 0;
  right: 0;
  z-index: 9;
  width: calc(100% - 210px);
}

.hideSidebar .fixed-header {
  width: calc(100% - 64px);
}

.fixed-header + .app-content {
  padding-top: 50px;
}

.hasTagsView {
  .app-content {
    /* 86 = navbar + tags-view = 50 + 36 */
    min-height: calc(100vh - 86px);
  }

  .fixed-header + .app-content {
    padding-top: 86px;
  }
}
</style>
