<template>
  <n-grid x-gap="24" y-gap="24" :cols="3">
    <n-gi>
      <div class="card-panel">
        <n-statistic label="USERS" tabular-nums>
          <n-number-animation
            show-separator
            :from="0"
            :to="1000"
          />
        </n-statistic>
      </div>
    </n-gi>
    <n-gi>
      <div class="card-panel">
        <n-statistic label="ITEMS" tabular-nums>
          <n-number-animation
            show-separator
            :from="0"
            :to="10000"
          />
        </n-statistic>
      </div>
    </n-gi>
    <n-gi>
      <div class="card-panel">
        <n-statistic label="TOTAL POSITIVE" tabular-nums>
          <n-number-animation
            show-separator
            :from="0"
            :to="100000"
          />
        </n-statistic>
      </div>
    </n-gi>
    <n-gi>
      <div class="card-panel">
        <n-statistic label="VALID POSITIVE" tabular-nums>
          <n-number-animation
            show-separator
            :from="0"
            :to="1000000"
          />
        </n-statistic>
      </div>
    </n-gi>
    <n-gi>
      <div class="card-panel">
        <n-statistic label="VALID NEGATIVE" tabular-nums>
          <n-number-animation
            show-separator
            :from="0"
            :to="10000000"
          />
        </n-statistic>
      </div>
    </n-gi>
  </n-grid>
</template>

<script lang="ts" setup>
import { ref } from 'vue'

const props = defineProps({})
</script>

<style lang="scss" scoped>
.card-panel {
  cursor: pointer;
  overflow: hidden;
  background: #fff;
  box-shadow: 4px 4px 40px rgb(0 0 0 / 5%);
  border-color: rgb(0 0 0 / 5%);
  padding: 24px;
  border-radius: 4px;
}
</style>
