<template>
  <n-breadcrumb class="app-breadcrumb">
    <transition-group name="breadcrumb">
      <n-breadcrumb-item
        v-for="(item, index) in levelList"
        :key="item.path"
        style="display: inline-block;"
      >
        <span
          v-if="item.redirect === 'noRedirect' || index === levelList.length - 1"
          class="no-redirect"
        >
          {{ item.meta.title }}
        </span>
        <a v-else @click.prevent="handleLink(item)">
          {{ item.meta.title }}
        </a>
      </n-breadcrumb-item>
    </transition-group>
  </n-breadcrumb>
</template>

<script lang="ts" setup>
import { ref, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const levelList = ref([])

const handleLink = item => {
  const { redirect, path } = item
  if (redirect) {
    router.push(redirect)
    return
  }
  router.push(path)
}

const isDashboard = route => {
  const name = route && route.name
  if (!name) {
    return false
  }

  return name.trim().toLocaleLowerCase() === 'Overview'.toLocaleLowerCase()
}

const getBreadcrumb = () => {
  // only show routes with meta.title
  let matched: any[] = route.matched.filter(item => item.meta && item.meta.title)
  const first = matched[1]

  if (!isDashboard(first)) {
    matched = [{ path: '/', meta: { title: 'Home' } }].concat(matched)
  }

  levelList.value = matched.filter(item => item.meta.breadcrumb !== false)
}

watch(route, () => {
  getBreadcrumb()
})

getBreadcrumb()
</script>

<style lang="scss" scoped>
.app-breadcrumb {
  display: inline-block;
  font-size: 14px;
  margin-left: 8px;
  height: 50px;
  line-height: 50px;

  .no-redirect {
    color: #97a8be;
    cursor: text;
  }
}
</style>
