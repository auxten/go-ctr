import { createApp } from 'vue'
import { createPinia } from 'pinia'

import NaiveUI from '@/components/naive-ui'
import '@/styles/index.scss'

import GUI from '@/components'

import router from './routes'
import App from './App.vue'

const app = createApp(App)
app.use(NaiveUI)
app.use(GUI)
app.use(createPinia())
app.use(router)

app.mount('#app')
