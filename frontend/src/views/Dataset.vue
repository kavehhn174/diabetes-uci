<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold">Dataset Showcase</h1>
      <p class="text-surface-500 dark:text-surface-400">
        Overview of the UCI Diabetes 130-US Hospitals dataset used to train the Bayesian Network.
      </p>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <ProgressSpinner />
    </div>

    <div v-else class="space-y-6">
      <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div class="card text-center">
          <div class="text-3xl font-bold text-primary-600 dark:text-primary-400">{{ stats.total?.toLocaleString() }}</div>
          <div class="text-sm text-surface-500 dark:text-surface-400">Total encounters</div>
        </div>
        <div class="card text-center">
          <div class="text-3xl font-bold text-amber-500">{{ (stats.readmissionRate * 100).toFixed(1) }}%</div>
          <div class="text-sm text-surface-500 dark:text-surface-400">Readmitted</div>
        </div>
        <div class="card text-center">
          <div class="text-3xl font-bold text-red-500">{{ (stats.within30Rate * 100).toFixed(1) }}%</div>
          <div class="text-sm text-surface-500 dark:text-surface-400">Readmitted &lt;30 days</div>
        </div>
        <div class="card text-center">
          <div class="text-3xl font-bold text-emerald-500">{{ stats.avgHospitalTime }}</div>
          <div class="text-sm text-surface-500 dark:text-surface-400">Avg. days in hospital</div>
        </div>
      </div>

      <div class="card grid gap-6 lg:grid-cols-2">
        <div>
          <h2 class="mb-2 text-lg font-bold">Readmission distribution</h2>
          <div class="h-72">
            <Bar :data="readmissionChart" :options="barOptions" />
          </div>
        </div>
        <div>
          <h2 class="mb-2 text-lg font-bold">Risk class distribution</h2>
          <div class="h-72">
            <Doughnut :data="riskChart" :options="doughnutOptions" />
          </div>
        </div>
      </div>

      <div class="card grid gap-6 lg:grid-cols-2">
        <div>
          <h2 class="mb-2 text-lg font-bold">Age distribution</h2>
          <div class="h-72">
            <Bar :data="ageChart" :options="barOptions" />
          </div>
        </div>
        <div>
          <h2 class="mb-2 text-lg font-bold">Readmission by age</h2>
          <div class="h-72">
            <Bar :data="ageReadmissionChart" :options="stackedBarOptions" />
          </div>
        </div>
      </div>

      <div class="card grid gap-6 lg:grid-cols-2">
        <div>
          <h2 class="mb-2 text-lg font-bold">Gender distribution</h2>
          <div class="h-64">
            <Pie :data="genderChart" :options="doughnutOptions" />
          </div>
        </div>
        <div>
          <h2 class="mb-2 text-lg font-bold">Race distribution</h2>
          <div class="h-64">
            <Pie :data="raceChart" :options="doughnutOptions" />
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="mb-2 text-lg font-bold">About the dataset</h2>
        <div class="space-y-3 text-surface-700 dark:text-surface-300">
          <p>
            The dataset contains <strong>101,766 diabetes encounters</strong> from 130 US hospitals over 10 years (1999–2008).
            Each row represents a hospital admission and includes demographic details, diagnosis codes, lab results,
            medications, and the target variable <code>readmitted</code>.
          </p>
          <p>
            The target has three classes:
            <Tag value="NO" severity="success" class="mx-1" /> not readmitted,
            <Tag value="&lt;30" severity="danger" class="mx-1" /> readmitted within 30 days, and
            <Tag value="&gt;30" severity="warn" class="mx-1" /> readmitted after 30 days.
            The 30-day readmission class is the most clinically critical and receives the highest weight in the risk score.
          </p>
          <p>
            A <strong>Bayesian Network</strong> is trained on the top features ranked by mutual information with the target.
            Structure learning discovers probabilistic dependencies among features, and the conditional probability tables
            are fitted via maximum likelihood estimation. The model is used to predict the probability of each readmission
            class for every encounter.
          </p>
          <p>
            Top predictive features include
            <strong>number of inpatient visits, discharge disposition, emergency visits, number of diagnoses,</strong>
            and <strong>medical specialty</strong>. These align with clinical intuition: patients with more prior hospital
            utilisation and complex discharge plans are more likely to be readmitted.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Bar, Doughnut, Pie } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale, ArcElement } from 'chart.js'
import { getStats } from '../services/api'

import ProgressSpinner from 'primevue/progressspinner'
import Tag from 'primevue/tag'

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale, ArcElement)

const stats = ref({})
const loading = ref(true)

const baseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'bottom' },
  },
}

const barOptions = {
  ...baseOptions,
  scales: { y: { beginAtZero: true } },
}

const stackedBarOptions = {
  ...baseOptions,
  scales: {
    x: { stacked: true },
    y: { stacked: true, beginAtZero: true },
  },
}

const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { position: 'bottom' } },
}

const readmissionChart = computed(() => ({
  labels: ['Not readmitted', '< 30 days', '> 30 days'],
  datasets: [{
    label: 'Encounters',
    data: [
      stats.value.total - stats.value.readmitted,
      stats.value.within30,
      stats.value.readmitted - stats.value.within30,
    ],
    backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
  }],
}))

const riskChart = computed(() => {
  const order = { low: 0, medium: 1, high: 2 }
  const ordered = [...(stats.value.riskDistribution || [])].sort((a, b) => order[a.risk_class] - order[b.risk_class])
  return {
    labels: ordered.map((d) => d.risk_class === 'low' ? 'Low' : d.risk_class === 'medium' ? 'Medium' : 'High'),
    datasets: [{
      data: ordered.map((d) => d.count),
      backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
    }],
  }
})

const ageChart = computed(() => ({
  labels: (stats.value.ageDistribution || []).map((d) => d.age),
  datasets: [{
    label: 'Patients',
    data: (stats.value.ageDistribution || []).map((d) => d.count),
    backgroundColor: '#3b82f6',
  }],
}))

const ageReadmissionChart = computed(() => {
  const byAge = {}
  for (const row of stats.value.readmittedByAge || []) {
    if (!byAge[row.age]) byAge[row.age] = {}
    byAge[row.age][row.readmitted] = row.count
  }
  const ages = Object.keys(byAge).sort((a, b) => a.localeCompare(b))
  return {
    labels: ages,
    datasets: [
      { label: 'NO', data: ages.map((a) => byAge[a]['NO'] || 0), backgroundColor: '#22c55e' },
      { label: '<30', data: ages.map((a) => byAge[a]['<30'] || 0), backgroundColor: '#ef4444' },
      { label: '>30', data: ages.map((a) => byAge[a]['>30'] || 0), backgroundColor: '#f59e0b' },
    ],
  }
})

const genderChart = computed(() => ({
  labels: (stats.value.genderDistribution || []).map((d) => d.gender),
  datasets: [{
    data: (stats.value.genderDistribution || []).map((d) => d.count),
    backgroundColor: ['#3b82f6', '#ec4899', '#6b7280'],
  }],
}))

const raceChart = computed(() => ({
  labels: (stats.value.raceDistribution || []).map((d) => d.race),
  datasets: [{
    data: (stats.value.raceDistribution || []).map((d) => d.count),
    backgroundColor: ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#6b7280'],
  }],
}))

onMounted(async () => {
  try {
    const { data } = await getStats()
    stats.value = data
  } catch (err) {
    console.error(err)
  } finally {
    loading.value = false
  }
})
</script>
