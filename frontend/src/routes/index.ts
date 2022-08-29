import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'
import { createDiscreteApi } from 'naive-ui'
import { globalConfig } from '@/config'
import { useUserStore } from '@/store/user'

const { loadingBar } = createDiscreteApi(['loadingBar'])

export const asyncRoutes: Array<RouteRecordRaw> = [
  {
    path: 'overview',
    name: 'Overview',
    // alias: ['/', '/home'],
    component: () => import('@/views/overview/index.vue'),
    meta: { title: 'Overview', affix: true },
  },
  {
    path: 'users',
    name: 'Users',
    component: () => import('@/views/users/index.vue'),
    meta: { title: 'Users' },
  },
  {
    path: 'items',
    name: 'Items',
    component: () => import('@/views/items/index.vue'),
    meta: { title: 'Items' },
  },
]

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    redirect: {
      name: 'Overview',
    },
  },
  {
    path: '/admin',
    redirect: 'noRedirect',
    component: () => import('@/layout/index.vue'),
    meta: { title: 'Admin', breadcrumb: false },
    children: [...asyncRoutes],
  },
  {
    path: '/login',
    component: () => import('@/views/login/index.vue'),
    meta: { title: 'Login' },
  },
  {
    path: '/:catchAll(.*)*',
    component: () => import('@/views/error/index.vue'),
    meta: { title: 'NotFound' },
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})



const whiteList = ['/login', '/auth-redirect', '/dev'] // no redirect whitelist

router.beforeEach(async (to, from, next) => {
  loadingBar.start()
  // set page title
  if (to.meta && to.meta.title) {
    document.title = `${to.meta.title} | ${globalConfig.title}`
  } else {
    document.title = globalConfig.title
  }

  const userStore = useUserStore()
  if (userStore.isLogin()) {
    if (to.path === '/login') {
      next({ path: '/' })
    } else {
      if (userStore.role > 0) {
        next()
      } else {
        try {
          await userStore.getUserInfo()
          next({ ...to, replace: true })
        } catch (error) {
          userStore.resetToken()
          next(`/login?redirect=${to.path}`)
        }
      }
    }
  } else if (whiteList.some(m => to.path.startsWith(m))) {
    next()
  } else {
    next(`/login?redirect=${to.path}`)
  }
})

router.afterEach(() => {
  loadingBar.finish()
})


export default router
