import request from '@/utils/request'

export function fetchOverview() {
  return request.get('/service/overview')
}

export function fetchUsers(page = 1, size = 20) {
  return request.get('/service/useritems', {
    params: {
      page,
      size,
    },
  })
}

export function fetchItems(page = 1, size = 20) {
  return request.get('/service/items', {
    params: {
      page,
      size,
    },
  })
}
