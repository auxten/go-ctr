<template>
  <div class="navbar">
    <hamburger
      id="hamburger-container"
      :is-active="appStore.sidebar.opened"
      class="hamburger-container"
      @toggleClick="toggleSideBar"
    />

    <breadcrumb id="breadcrumb-container" class="breadcrumb-container" />

    <div class="right-menu">
      <n-dropdown
        :options="menus"
        placement="bottom-end"
        trigger="click"
        show-arrow
      >
        <div class="avatar-wrapper">
          <img src="../assets/avatar.gif" class="user-avatar">
          <n-icon class="caret-bottom">
            <CaretDownOutline />
          </n-icon>
        </div>
      </n-dropdown>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { h } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useAppStore } from '@/store/app'
import { useUserStore } from '@/store/user'
import { CaretDownOutline } from '@vicons/ionicons5'
import hamburger from './components/hamburger.vue'
import breadcrumb from './components/breadcrumb.vue'

const router = useRouter()
const appStore = useAppStore()
const userStore = useUserStore()

const menus = [
  {
    key: 'Overview',
    label: () =>
      h(
        RouterLink,
        {
          to: {
            name: 'Home',
          },
        },
        { default: () => 'Home' },
      ),
  },
  {
    key: 'Github',
    label: () =>
      h(
        'a',
        {
          href: 'https://github.com/auxten/edgeRec',
          target: '_blank',
          rel: 'noopenner noreferrer',
        },
        'Github',
      ),
  },
  {
    type: 'divider',
    key: 'd1',
  },
  {
    label: 'Log Out',
    key: 'logout',
    props: {
      onClick: () => {
        userStore.logout()
        router.push(`/login?redirect=${router.currentRoute.value.fullPath}`)
      },
    },
  },
]

const toggleSideBar = () => {
  appStore.toggleSidebar()
}
</script>

<style lang="scss" scoped>
.navbar {
  height: 50px;
  overflow: hidden;
  position: relative;
  background: #fff;
  box-shadow: 0 1px 4px rgb(0 21 41 / 8%);

  .hamburger-container {
    line-height: 46px;
    height: 100%;
    float: left;
    cursor: pointer;
    transition: background 0.3s;
    -webkit-tap-highlight-color: transparent;

    &:hover {
      background: rgb(0 0 0 / 2.5%);
    }
  }

  .errLog-container {
    display: inline-block;
    vertical-align: top;
  }

  .right-menu {
    float: right;
    height: 100%;
    line-height: 50px;

    &:focus {
      outline: none;
    }

    .right-menu-item {
      display: inline-block;
      padding: 0 8px;
      height: 100%;
      font-size: 18px;
      color: #5a5e66;
      vertical-align: text-bottom;

      &.hover-effect {
        cursor: pointer;
        transition: background 0.3s;

        &:hover {
          background: rgb(0 0 0 / 2.5%);
        }
      }
    }

    .avatar-wrapper {
      margin-top: 5px;
      position: relative;
      margin-right: 30px;
      height: 50px;

      .user-avatar {
        cursor: pointer;
        width: 40px;
        height: 40px;
        border-radius: 10px;
      }

      .caret-bottom {
        cursor: pointer;
        position: absolute;
        right: -20px;
        top: 25px;
        font-size: 12px;
      }
    }
  }
}
</style>
