import { defineStore } from 'pinia'

const TokenKey = 'ER-Token'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: '',
    name: '',
    avatar: '',
    role: -1,
  }),
  actions: {
    async login(username: string, password: string) {
      const time = Date.now()
      const token = window.btoa(`${time},${username}${password}`)
      this.token = token
      localStorage.setItem(TokenKey, token)

      this.name = username
    },
    async getUserInfo() {
      this.role = 1
    },
    resetToken() {
      this.token = ''
      localStorage.removeItem(TokenKey)
    },
    isLogin() {
      const token = localStorage.getItem(TokenKey)
      if (token) {
        const time = parseInt(window.atob(token))
        const diff = Date.now() - time
        return diff < 3 * 24 * 60 * 60 * 1000
      }
      return false
    },
    async logout() {
      this.resetToken()
    },
  },
})
