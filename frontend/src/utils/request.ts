import axios from 'axios'

// create an axios instance
const service = axios.create({
  baseURL: '/',
  timeout: 30000,
  withCredentials: true,
})

// request interceptor
service.interceptors.request.use(
  config => {
    return config
  },

  error => {
    // do something with request error
    console.error(error) // for debug
    return Promise.reject(error)
  },
)

// response interceptor
service.interceptors.response.use(
  response => {
    return response
  },
  error => {
    return Promise.reject(error)
  },
)

export default service
