import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'
import { globalConfig } from '@/config'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    redirect: {
      name: 'Overview',
    },
  },
  {
    path: '/overview',
    name: 'Overview',
    // alias: ['/', '/home'],
    component: () => import('@/views/overview/index.vue'),
    meta: { title: 'Dashboard' },
  },
  {
    path: '/users',
    name: 'Users',
    component: () => import('@/views/users/index.vue'),
    meta: { title: 'Users' },
  },
  {
    path: '/items',
    name: 'Items',
    component: () => import('@/views/items/index.vue'),
    meta: { title: 'Items' },
  },
  {
    path: '/:catchAll(.*)*',
    name: 'NotFound',
    component: () => import('@/views/error/index.vue'),
    meta: { title: 'NotFound' },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})


router.beforeEach(async (to, from, next) => {
  // set page title
  if (to.meta && to.meta.title) {
    document.title = `${to.meta.title} | ${globalConfig.title}`
  } else {
    document.title = globalConfig.title
  }

  next()
})


export default router
