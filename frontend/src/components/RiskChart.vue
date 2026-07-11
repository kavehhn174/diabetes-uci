<template>
  <div class="h-64">
    <Doughnut :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Doughnut } from 'vue-chartjs'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend)

const props = defineProps({
  patient: { type: Object, required: true },
})

const chartData = computed(() => ({
  labels: ['No readmission', 'Readmitted <30 days', 'Readmitted >30 days'],
  datasets: [
    {
      data: [
        props.patient.prob_no ?? 0,
        props.patient.prob_lt30 ?? 0,
        props.patient.prob_gt30 ?? 0,
      ],
      backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
      borderWidth: 0,
    },
  ],
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        usePointStyle: true,
        padding: 16,
      },
    },
    tooltip: {
      callbacks: {
        label: (ctx) => {
          const value = (ctx.raw * 100).toFixed(1)
          return `${ctx.label}: ${value}%`
        },
      },
    },
  },
}
</script>
