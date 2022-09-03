<template>
  <n-space vertical :size="24">
    <n-grid x-gap="24" y-gap="24" :cols="3">
      <n-gi>
        <n-card hoverable>
          <n-statistic label="USERS" tabular-nums>
            <n-number-animation
              show-separator
              :from="0"
              :to="1000"
            />
          </n-statistic>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card hoverable>
          <n-statistic label="ITEMS" tabular-nums>
            <n-number-animation
              show-separator
              :from="0"
              :to="10000"
            />
          </n-statistic>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card hoverable>
          <n-statistic label="TOTAL POSITIVE" tabular-nums>
            <n-number-animation
              show-separator
              :from="0"
              :to="100000"
            />
          </n-statistic>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card hoverable>
          <n-statistic label="VALID POSITIVE" tabular-nums>
            <n-number-animation
              show-separator
              :from="0"
              :to="1000000"
            />
          </n-statistic>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card hoverable>
          <n-statistic label="VALID NEGATIVE" tabular-nums>
            <n-number-animation
              show-separator
              :from="0"
              :to="10000000"
            />
          </n-statistic>
        </n-card>
      </n-gi>
    </n-grid>
    <n-card
      title="Positive Feedback Rate"
      :segmented="{
        content: true,
      }"
      hoverable
      content-style="padding: 0;"
    >
      <div :id="chartId" class="chart-box"></div>
    </n-card>
  </n-space>
</template>

<script lang="ts" setup>
import { onMounted, shallowRef } from 'vue'
import * as echarts from 'echarts/core'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { UniversalTransition } from 'echarts/features'
import { debounce } from 'lodash-es'
import { off, on } from '@/utils/dom-util'

echarts.use([
  GridComponent,
  LegendComponent,
  TooltipComponent,
  LineChart,
  CanvasRenderer,
  UniversalTransition,
])

const chartId = 'myChart'
const chartIns = shallowRef<echarts.ECharts | null>(null)

const initChart = (data1: any[], data2: any[]) => {
  const chartDom = document.getElementById(chartId)
  const chart = echarts.init(chartDom)
  chart.setOption({
    grid: {
      top: 36,
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
    },
    legend: {
      data: ['like', 'star'],
    },
    xAxis: {
      type: 'category',
      // name: 'like',
      // nameLocation: 'middle',
      // nameGap: 30,
      // nameTextStyle: {
      //   color: '#636366',
      // },
      boundaryGap: false,
      axisLabel: {
        color: '#666',
        rotate: 45,
        // formatter(text) {
        //   return parseFloat(text).toFixed(1)
        // },
      },
      axisLine: {
        show: true,
        lineStyle: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    },
    yAxis: {
      type: 'value',
      // name: '',
      // nameLocation: 'middle',
      // nameGap: 40,
      // nameTextStyle: {
      //   color: '#636366',
      // },
      axisLabel: {
        color: '#666',
      },
      axisLine: {
        show: true,
        lineStyle: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
    series: [
      {
        name: 'like',
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: {
          color: '#ff6384',
        },
        itemStyle: {
          color: '#ff6384',
        },
        data: [120, 132, 101, 134, 90, 230, 210],
      },
      {
        name: 'star',
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: {
          color: '#36a2eb',
        },
        itemStyle: {
          color: '#36a2eb',
        },
        data: [220, 182, 191, 234, 290, 330, 310],
      },
    ],
  })

  chartIns.value = chart
}

const resizeChart = debounce(() => {
  chartIns.value?.resize()
}, 300)

onMounted(() => {
  off(window, 'resize', resizeChart)
  on(window, 'resize', resizeChart)

  initChart([], [])
})
</script>

<style lang="scss" scoped>
.chart-box {
  width: 100%;
  height: 350px;
  margin: 12px auto 0;
}
</style>
