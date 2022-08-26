import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'
import { globalConfig } from '@/config'
import { useUserStore } from '@/store/user'

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
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login/index.vue'),
    meta: { title: 'Login' },
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



const whiteList = ['/login', '/auth-redirect', '/dev'] // no redirect whitelist

router.beforeEach(async (to, from, next) => {
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


export default router
