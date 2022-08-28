<template>
  <div id="tags-view-container" class="tags-view-container">
    <div ref="scrollPane" class="tags-view-wrapper">
      <router-link
        v-for="tag in visitedViews"
        :key="tag.path"
        v-slot="{ navigate }"
        :to="{
          path: tag.path,
          query: tag.query,
        }"
        custom
      >
        <span
          :class="isActive(tag)?'active':''"
          class="tags-view-item"
          @click="navigate"
        >
          {{ tag.title }}
          <n-icon
            v-if="!isAffix(tag)"
            class="icon-btn-close"
            @click.prevent.stop="closeSelectedTag(tag)"
          >
            <CloseOutline />
          </n-icon>
        </span>
      </router-link>
    </div>
    <ul v-show="visible" :style="{left:left+'px',top:top+'px'}" class="contextmenu">
      <li @click="refreshSelectedTag(selectedTag)">Refresh</li>
      <li v-if="!isAffix(selectedTag)" @click="closeSelectedTag(selectedTag)">Close</li>
      <li @click="closeOthersTags">Close Others</li>
      <li @click="closeAllTags(selectedTag)">Close All</li>
    </ul>
  </div>
</template>

<script lang="ts" setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { CloseOutline } from '@vicons/ionicons5'
import { useTagsViewStore } from '@/store/tags-view'
import { asyncRoutes } from '@/routes'

const route = useRoute()
const router = useRouter()
const tagsViewStore = useTagsViewStore()

const visible = ref(false)
const top = ref(0)
const left = ref(0)
const selectedTag = ref({})
const affixTags = ref([])

const visitedViews = computed(() => {
  return tagsViewStore.visitedViews
})

const isAffix = tag => {
  return tag.meta && tag.meta.affix
}

const isActive = cur => {
  return cur.path === route.path
}

const filterAffixTags = routes => {
  let tags = []
  routes.forEach(route => {
    if (route.meta && route.meta.affix) {
      const tag = router.resolve(route)
      tags.push({
        fullPath: tag.fullPath,
        path: tag.path,
        name: route.name,
        meta: { ...route.meta },
      })
    }
  })
  return tags
}

const initTags = () => {
  affixTags.value = filterAffixTags(asyncRoutes)
  for (const tag of affixTags.value ) {
    // Must have tag name
    if (tag.name) {
      tagsViewStore.addVisitedView(tag)
    }
  }
}

const addTags = () => {
  if (route.name) {
    tagsViewStore.addView(route)
  }
  return false
}

const refreshSelectedTag = view => {
  tagsViewStore.delCachedView(view).then(() => {
    const { fullPath } = view
    nextTick(() => {
      router.replace({
        path: `/redirect${fullPath}`,
      })
    })
  })
}

const closeSelectedTag = view => {
  tagsViewStore.delView(view).then(data => {
    if (isActive(view)) {
      toLastView(data.visitedViews, view)
    }
  })
}

const closeOthersTags = () => {
  router.push(selectedTag.value)
  tagsViewStore.delOthersViews(selectedTag.value)
    .then(() => {})
}

const closeAllTags = view => {
  tagsViewStore.delAllViews().then(data => {
    if (affixTags.value.some(tag => tag.path === view.path)) {
      return
    }
    toLastView(data.visitedViews, view)
  })
}

const toLastView = (visitedViews, view) => {
  const latestView = visitedViews.slice(-1)[0]
  if (latestView) {
    router.push(latestView.fullPath)
  } else {
    // now the default is to redirect to the home page if there is no tags-view,
    // you can adjust it according to your needs.
    if (view.name === 'Overview') {
      // to reload home page
      router.replace({ path: `/redirect${  view.fullPath}` })
    } else {
      router.push('/')
    }
  }
}

const closeMenu = () => {
  visible.value = false
}

watch(route, () => {
  addTags()
})

watch(visible, val => {
  if (val) {
    document.body.addEventListener('click', closeMenu)
  } else {
    document.body.removeEventListener('click', closeMenu)
  }
})

onMounted(() => {
  initTags()
  addTags()
})


</script>


<style lang="scss" scoped>
.tags-view-container {
  height: 36px;
  width: 100%;
  background: #fff;
  border-bottom: 1px solid #d8dce5;
  box-shadow: 0 1px 3px 0 rgb(0 0 0 / 12%), 0 0 3px 0 rgb(0 0 0 / 4%);

  .tags-view-wrapper {
    .tags-view-item {
      display: inline-block;
      position: relative;
      cursor: pointer;
      height: 28px;
      line-height: 26px;
      border: 1px solid #d8dce5;
      color: #495060;
      background: #fff;
      padding: 0 8px;
      font-size: 12px;
      margin-left: 5px;
      margin-top: 4px;

      &:first-of-type {
        margin-left: 15px;
      }

      &:last-of-type {
        margin-right: 15px;
      }

      &.active {
        background-color: #42b983;
        color: #fff;
        border-color: #42b983;

        &::before {
          content: '';
          background: #fff;
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          position: relative;
          margin-right: 2px;
        }
      }
    }
  }

  .contextmenu {
    margin: 0;
    background: #fff;
    z-index: 3000;
    position: absolute;
    list-style-type: none;
    padding: 5px 0;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 400;
    color: #333;
    box-shadow: 2px 2px 3px 0 rgb(0 0 0 / 30%);

    li {
      margin: 0;
      padding: 7px 16px;
      cursor: pointer;

      &:hover {
        background: #eee;
      }
    }
  }
}
</style>

<style lang="scss">
.tags-view-wrapper {
  .tags-view-item {
    .icon-btn-close {
      border-radius: 50%;
      text-align: center;
      display: inline-block;
      vertical-align: -3px;
      transition: all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1);
      transform-origin: 100% 50%;

      &:hover {
        background-color: #b4bccc;
        color: #fff;
      }
    }
  }
}
</style>

