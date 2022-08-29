<template>
  <div class="drawer-container">
    <div>
      <h3 class="drawer-title">Page style setting</h3>

      <div class="drawer-item">
        <span>Open Tags-View</span>
        <n-switch v-model:value="tagsView" class="drawer-switch" />
      </div>

      <div class="drawer-item">
        <span>Fixed Header</span>
        <n-switch v-model:value="fixedHeader" class="drawer-switch" />
      </div>

      <div class="drawer-item">
        <span>Sidebar Logo</span>
        <n-switch v-model:value="sidebarLogo" class="drawer-switch" />
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed } from 'vue'
import { useSettingStore } from '@/store/setting'


const settingStore = useSettingStore()

const tagsView = computed({
  get() {
    return settingStore.tagsView
  },
  set(val){
    settingStore.changeSetting('tagsView', val)
  },
})

const fixedHeader = computed({
  get() {
    return settingStore.fixedHeader
  },
  set(val){
    settingStore.changeSetting('fixedHeader', val)
  },
})

const sidebarLogo = computed({
  get() {
    return settingStore.sidebarLogo
  },
  set(val){
    settingStore.changeSetting('sidebarLogo', val)
  },
})
</script>

<style lang="scss" scoped>
.drawer-container {
  padding: 24px;
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;

  .drawer-title {
    margin-bottom: 12px;
    color: rgb(0 0 0 / 85%);
    font-size: 14px;
    line-height: 22px;
  }

  .drawer-item {
    color: rgb(0 0 0 / 65%);
    font-size: 14px;
    padding: 12px 0;
  }

  .drawer-switch {
    float: right;
  }
}
</style>

