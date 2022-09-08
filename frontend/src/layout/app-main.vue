<template>
  <section class="app-main">
    <router-view v-slot="{ Component }">
      <transition name="fade-transform" mode="out-in">
        <keep-alive :include="cachedViews">
          <component :is="Component" />
        </keep-alive>
      </transition>
    </router-view>
  </section>
</template>


<script lang="ts" setup>
import { computed } from 'vue'
import { useTagsViewStore } from '@/store/tags-view'

const tagsViewStore = useTagsViewStore()

const cachedViews = computed(() => {
  return tagsViewStore.cachedViews
})
</script>


<style lang="scss" scoped>
.app-main {
  min-height: calc(100vh - 86px);
  width: 100%;
  position: relative;
  overflow: hidden;
  background-color: #f0f2f5;
  padding: 16px 24px 20px;
}
</style>

