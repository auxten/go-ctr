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
    },
    async getUserInfo() {
    },
    resetToken() {
      this.token = ''
      localStorage.removeItem(TokenKey)
    },
    isLogin() {
      const token = localStorage.getItem(TokenKey)
      return !!token
    },
    async logout() {
    },
  },
})
