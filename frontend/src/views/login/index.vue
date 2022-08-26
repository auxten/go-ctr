<template>
  <div class="login-container">
    <n-form
      ref="loginFormRef"
      :model="loginForm"
      :rules="loginRules"
      label-placement="left"
      size="small"
      class="login-form"
    >
      <div class="logo-wrap">
        <img src="../../assets/avatar.webp">
      </div>

      <n-h2 strong style="--n-margin: 20px 0 16px 0;">Sign In</n-h2>
      <n-divider style="margin: 0 0 16px;" />

      <n-form-item path="username">
        <n-input
          v-model:value="loginForm.username"
          placeholder="UserName"
          type="text"
          size="large"
        >
          <template #prefix>
            <n-icon>
              <PersonOutline />
            </n-icon>
          </template>
        </n-input>
      </n-form-item>
      <n-form-item path="password">
        <n-tooltip :show="capsTooltip" placement="top-start">
          <template #trigger>
            <n-input
              v-model:value="loginForm.password"
              placeholder="Password"
              type="password"
              size="large"
              show-password-on="click"
              @keydown="checkCapslock"
              @blur="capsTooltip = false"
              @keyup.enter="handleLogin"
            >
              <template #prefix>
                <n-icon>
                  <KeyOutline />
                </n-icon>
              </template>
            </n-input>
          </template>
          <span> Capital Lock ON </span>
        </n-tooltip>
      </n-form-item>
      <n-button
        :loading="loading"
        type="info"
        size="large"
        style="width: 100%; margin-bottom: 20px;"
        @click="handleLogin"
      >
        Sign In
      </n-button>
    </n-form>
  </div>
</template>

<script lang="ts" setup>
import { ref, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { PersonOutline, KeyOutline } from '@vicons/ionicons5'
import { useUserStore } from '@/store/user'

const validateUsername = (rule: any, value: string, callback: Function) => {
  if (!['admin', 'editor'].includes(value)) {
    callback(new Error('请输入正确的用户名'))
  } else {
    callback()
  }
}

const validatePassword = (rule: any, value: string, callback: Function) => {
  if (value.length < 6) {
    callback(new Error('密码不能少于6位'))
  } else {
    callback()
  }
}

const getOtherQuery = (query: any) => {
  return Object.keys(query).reduce((acc: any, cur) => {
    if (cur !== 'redirect') {
      acc[cur] = query[cur]
    }
    return acc
  }, {})
}

const userStore = useUserStore()

const loginForm = ref({
  username: 'admin',
  password: '123123',
})

const loginRules = ref({
  username: [{ required: true, trigger: 'blur', validator: validateUsername }],
  password: [{ required: true, trigger: 'blur', validator: validatePassword }],
})

const loginFormRef = ref(null)
const capsTooltip = ref(false)
const loading = ref(false)
const redirect = ref('')
const otherQuery = ref({})

const route = useRoute()
const router = useRouter()

watch(route, ({ query }) => {
  if (query) {
    redirect.value = query.redirect as string
    otherQuery.value = getOtherQuery(query)
  }
}, { immediate: true })

const checkCapslock = ({ shiftKey, key }: any) => {
  if (key && key.length === 1) {
    if (shiftKey && (key >= 'a' && key <= 'z') || !shiftKey && (key >= 'A' && key <= 'Z')) {
      capsTooltip.value = true
    } else {
      capsTooltip.value = false
    }
  }

  if (key === 'CapsLock' && capsTooltip.value === true) {
    capsTooltip.value = false
  }
}

const handleLogin = () => {
  (loginFormRef.value as any).validate((errors: any) => {
    if (!errors) {
      loading.value = true
      userStore.login(loginForm.value.username, loginForm.value.password)
        .then(() => {
          router.push({ path: redirect.value || '/', query: otherQuery.value })
        })
        .finally(() => {
          loading.value = false
        })
    }
  })
}
</script>

<style lang="scss" scoped>
.login-container {
  width: 100%;
  min-height: 100%;
  height: 100vh;
  overflow: hidden;
  background-color: #f7f7f7;

  .login-form {
    position: relative;
    width: 480px;
    max-width: 100%;
    margin: 0 auto;
    overflow: hidden;
    box-shadow: 0 10px 20px rgb(0 0 0 / 20%);
    padding: 40px;
    margin-top: 120px;
    border-radius: 10px;
    background-color: #fff;

    .logo-wrap {
      text-align: center;

      > img {
        width: 120px;
        height: auto;
      }
    }
  }
}
</style>
