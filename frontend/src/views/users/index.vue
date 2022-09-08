<template>
  <n-card
    title="Users"
    size="small"
    :segmented="{ content: true }"
  >
    <!-- <n-input
      v-model:value="searchText"
      placeholder="Search"
      style="margin-bottom: 12px;"
      @change="onSearch"
    >
      <template #prefix>
        <n-icon :component="SearchOutline" />
      </template>
    </n-input> -->

    <n-data-table
      :loading="loading"
      :columns="columns"
      :data="dataList"
      remote
      :pagination="pagination"
      max-height="calc(100vh - 180px)"
      v-bind="tableAttr"
      @update-page="handleUpdatePage"
    />
  </n-card>
</template>

<script lang="ts" setup>
import { computed, h, onMounted, reactive, ref } from 'vue'
import { DataTableColumn, NEllipsis, NTag, useMessage, NSpace } from 'naive-ui'
import { fetchOverview, fetchUsers } from '@/api'
// import { useRouter } from 'vue-router'
// import { SearchOutline } from '@vicons/ionicons5'

// const router = useRouter()
const nMessage = useMessage()

const loading = ref(false)
// const searchText = ref('')

const cols = ref([])
const dataList = ref([])

const pagination = reactive({
  page: 1,
  itemCount: 0,
  pageSize: 20,
})

const createColumns = () => {
  const list: DataTableColumn[] = []
  cols.value.forEach(item => {
    list.push({
      key: item.col,
      title: item.col,
      width: 200,
      render: (row: any) => {
        const data = row[item.col]
        if (Array.isArray(data)) {
          return h(NSpace, {
            size: 6,
          }, {
            default: () => data.map(m => {
              return h(
                NTag,
                {
                  type: 'info',
                  bordered: false,
                },
                {
                  default: () => m,
                },
              )
            }),
          })
        }
        return h(NEllipsis, {
          tooltip: true,
        }, () => `${data}`)
      },
    })
  })
  // list.push({
  //   title: 'Action',
  //   key: 'actions',
  //   width: 100,
  //   render (row) {
  //     return h(
  //       NButton,
  //       {
  //         size: 'small',
  //         type: 'info',
  //         ghost: true,
  //         onClick: () => {
  //           console.log(row)
  //           router.push({
  //             name: 'UserDetail',
  //             params: { id: '1' },
  //           })
  //         },
  //       },
  //       { default: () => 'Insight' },
  //     )
  //   },
  // })
  return list
}

const columns = computed(() => {
  return createColumns()
})

const tableAttr = computed(() => {
  const len = columns.value.length
  return {
    scrollX: len * 200,
    rowKey: (row: any) => row.item_id,
  }
})

const handleUpdatePage = (page: number) => {
  pagination.page = page
  loadData()
}

// const onSearch = () => {
//   pagination.page = 1
//   loadData()
// }

const getOverview = async () => {
  try {
    const res = await fetchOverview()
    if (res.data) {
      pagination.itemCount = res.data.users
    }
  } catch (error) {
    nMessage.error(error.toString())
  }
}

const loadData = async () => {
  try {
    loading.value = true
    const res = await fetchUsers(pagination.page, pagination.pageSize)
    if (res.data.users) {
      dataList.value = res.data.users.map(m => ({
        user_id: m.user_id,
        ...m.UserFeatures,
      }))

      cols.value = [{ col: 'user_id' }]
      cols.value.push(...Object.keys(res.data.users[0].UserFeatures).map(m => ({ col: m })))
    }
  } catch (error) {
    nMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  getOverview()
  loadData()
})
</script>

<script lang="ts">
export default {
  name: 'UserList',
}
</script>
