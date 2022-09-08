<template>
  <div>user-detail</div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useTagsViewStore } from '@/store/tags-view'

const props = defineProps({
  id: {
    type: String,
    required: true,
  },
})

const route = useRoute()
const tagsViewStore = useTagsViewStore()

const tempRoute = Object.assign({}, route)

const setTagsViewTitle = () => {
  const title = 'User Detail'
  const newRoute = Object.assign(
    {},
    tempRoute,
    { title: `${title}-${props.id}` },
  )
  tagsViewStore.updateVisitedView(newRoute)
}

onMounted(() => {
  setTagsViewTitle()
})
</script>

<style lang="scss" scoped>
</style>
