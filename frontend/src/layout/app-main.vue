<template>
  <section class="app-main">
    <router-view v-slot="{ Component }" :key="key">
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
import { useRoute } from 'vue-router'
import { useTagsViewStore } from '@/store/tags-view'

const route = useRoute()
const tagsViewStore = useTagsViewStore()

const cachedViews = computed(() => {
  return tagsViewStore.cachedViews
})
const key = computed(() => route.path)
</script>


<style lang="scss" scoped>
.app-main {
  min-height: calc(100vh - 126px);
  width: 100%;
  position: relative;
  overflow: hidden;
}
</style>

