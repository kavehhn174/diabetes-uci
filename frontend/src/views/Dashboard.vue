<template>
  <div class="space-y-6">
    <div class="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h1 class="text-2xl font-bold">Patient Readmission Risk</h1>
        <p class="text-surface-500 dark:text-surface-400">
          Query the diabetic patient dataset and inspect Bayesian-network risk predictions.
        </p>
      </div>
      <div class="flex items-center gap-2">
        <Tag v-if="usingModel" value="Model predictions" severity="success" icon="pi pi-check-circle" />
        <Tag v-else value="Heuristic fallback" severity="warn" icon="pi pi-exclamation-circle" />
      </div>
    </div>

    <div class="card grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Age</label>
        <MultiSelect
          v-model="filters.age"
          :options="ageOptions"
          optionLabel="label"
          optionValue="value"
          placeholder="Select ages"
          class="w-full"
          display="chip"
        />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Gender</label>
        <MultiSelect
          v-model="filters.gender"
          :options="genderOptions"
          optionLabel="label"
          optionValue="value"
          placeholder="Select genders"
          class="w-full"
          display="chip"
        />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Risk class</label>
        <MultiSelect
          v-model="filters.risk_class"
          :options="riskOptions"
          optionLabel="label"
          optionValue="value"
          placeholder="Select risk"
          class="w-full"
          display="chip"
        />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Readmitted</label>
        <MultiSelect
          v-model="filters.readmitted"
          :options="readmittedOptions"
          optionLabel="label"
          optionValue="value"
          placeholder="Select status"
          class="w-full"
          display="chip"
        />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Race</label>
        <MultiSelect
          v-model="filters.race"
          :options="raceOptions"
          optionLabel="label"
          optionValue="value"
          placeholder="Select races"
          class="w-full"
          display="chip"
        />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Medical specialty</label>
        <InputText v-model="filters.medical_specialty" placeholder="e.g. Cardiology" class="w-full" />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-sm font-medium text-surface-500 dark:text-surface-400">Min risk score</label>
        <Slider v-model="minRisk" :min="0" :max="1" :step="0.01" class="w-full" />
        <span class="text-xs text-surface-500">{{ minRisk.toFixed(2) }}</span>
      </div>
      <div class="flex items-end gap-2">
        <Button label="Search" icon="pi pi-search" @click="loadPatients" />
        <Button label="Clear" icon="pi pi-refresh" severity="secondary" @click="clearFilters" />
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-3">
      <div class="card lg:col-span-2">
        <DataTable
          :value="patients"
          :loading="loading"
          v-model:selection="selectedRow"
          paginator
          :rows="limit"
          :totalRecords="total"
          :rowsPerPageOptions="[10, 20, 50]"
          lazy
          @page="onPageChange"
          @update:rows="onRowsChange"
          selectionMode="single"
          dataKey="encounter_id"
          @rowSelect="onRowSelect"
          class="p-datatable-sm"
          scrollable
          scrollHeight="500px"
        >
          <Column field="encounter_id" header="Encounter ID" sortable />
          <Column field="age" header="Age" sortable />
          <Column field="gender" header="Gender" sortable />
          <Column field="race" header="Race" sortable />
          <Column field="risk_score" header="Risk" sortable>
            <template #body="{ data }">
              <div class="flex items-center gap-3">
                <ProgressBar
                  :value="data.risk_score * 100"
                  :showValue="false"
                  :class="riskBarClass(data.risk_class)"
                  class="h-3 w-28"
                />
                <span class="min-w-12 text-right text-sm font-semibold tabular-nums">
                  {{ (data.risk_score * 100).toFixed(0) }}%
                </span>
              </div>
            </template>
          </Column>
          <Column field="risk_label" header="Class" sortable>
            <template #body="{ data }">
              <Tag :value="data.risk_label" :severity="riskSeverity(data.risk_class)" />
            </template>
          </Column>
          <Column field="readmitted" header="Actual" sortable>
            <template #body="{ data }">
              <Tag :value="data.readmitted" :severity="readmittedSeverity(data.readmitted)" />
            </template>
          </Column>
        </DataTable>
      </div>

      <div class="card">
        <div v-if="selectedPatient" class="space-y-5">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold">Patient {{ selectedPatient.encounter_id }}</h2>
            <Tag :value="selectedPatient.risk_label" :severity="riskSeverity(selectedPatient.risk_class)" />
          </div>

          <div class="rounded-xl border border-surface-200 p-4 dark:border-surface-700">
            <RiskChart :patient="selectedPatient" />
          </div>

          <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="rounded-lg bg-surface-100 p-3 dark:bg-surface-700">
              <div class="text-surface-500 dark:text-surface-400">Age</div>
              <div class="font-semibold">{{ selectedPatient.age }}</div>
            </div>
            <div class="rounded-lg bg-surface-100 p-3 dark:bg-surface-700">
              <div class="text-surface-500 dark:text-surface-400">Gender</div>
              <div class="font-semibold">{{ selectedPatient.gender }}</div>
            </div>
            <div class="rounded-lg bg-surface-100 p-3 dark:bg-surface-700">
              <div class="text-surface-500 dark:text-surface-400">Race</div>
              <div class="font-semibold">{{ selectedPatient.race }}</div>
            </div>
            <div class="rounded-lg bg-surface-100 p-3 dark:bg-surface-700">
              <div class="text-surface-500 dark:text-surface-400">Specialty</div>
              <div class="font-semibold truncate" :title="selectedPatient.medical_specialty">
                {{ selectedPatient.medical_specialty || 'N/A' }}
              </div>
            </div>
          </div>

          <div class="space-y-2">
            <h3 class="text-sm font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-400">Top risk factors</h3>
            <div class="space-y-1 text-sm">
              <div class="flex justify-between">
                <span>Inpatient visits</span>
                <span class="font-semibold">{{ selectedPatient.number_inpatient }}</span>
              </div>
              <div class="flex justify-between">
                <span>Emergency visits</span>
                <span class="font-semibold">{{ selectedPatient.number_emergency }}</span>
              </div>
              <div class="flex justify-between">
                <span>Diagnoses</span>
                <span class="font-semibold">{{ selectedPatient.number_diagnoses }}</span>
              </div>
              <div class="flex justify-between">
                <span>Lab procedures</span>
                <span class="font-semibold">{{ selectedPatient.num_lab_procedures }}</span>
              </div>
              <div class="flex justify-between">
                <span>Time in hospital</span>
                <span class="font-semibold">{{ selectedPatient.time_in_hospital }} days</span>
              </div>
            </div>
          </div>

          <div class="text-xs text-surface-500 dark:text-surface-400">
            Predicted class: <strong>{{ selectedPatient.predicted_class }}</strong> ·
            Actual: <strong>{{ selectedPatient.readmitted }}</strong>
          </div>
        </div>
        <div v-else class="flex h-64 flex-col items-center justify-center text-surface-400">
          <i class="pi pi-user mb-2 text-4xl"></i>
          <p>Select a patient to view risk details</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { searchPatients, predictPatient, getStats } from '../services/api'

import Button from 'primevue/button'
import Column from 'primevue/column'
import DataTable from 'primevue/datatable'
import InputText from 'primevue/inputtext'
import MultiSelect from 'primevue/multiselect'
import ProgressBar from 'primevue/progressbar'
import Slider from 'primevue/slider'
import Tag from 'primevue/tag'
import RiskChart from '../components/RiskChart.vue'

const filters = ref({
  age: [],
  gender: [],
  risk_class: [],
  readmitted: [],
  race: [],
  medical_specialty: '',
})
const minRisk = ref(0)

const ageOptions = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'].map((a) => ({ label: a, value: a }))
const genderOptions = ['Male', 'Female', 'Unknown/Invalid'].map((g) => ({ label: g, value: g }))
const riskOptions = ['low', 'medium', 'high'].map((r) => ({ label: r === 'low' ? 'Low' : r === 'medium' ? 'Medium' : 'High', value: r }))
const readmittedOptions = ['NO', '<30', '>30'].map((r) => ({ label: r === 'NO' ? 'Not readmitted' : r === '<30' ? '< 30 days' : '> 30 days', value: r }))
const raceOptions = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'].map((r) => ({ label: r, value: r }))

const patients = ref([])
const total = ref(0)
const page = ref(1)
const limit = ref(20)
const loading = ref(false)
const selectedRow = ref(null)
const selectedPatient = ref(null)
const usingModel = ref(false)

function buildFilterPayload() {
  const payload = {}
  if (filters.value.age.length) payload.age = filters.value.age
  if (filters.value.gender.length) payload.gender = filters.value.gender
  if (filters.value.risk_class.length) payload.risk_class = filters.value.risk_class
  if (filters.value.readmitted.length) payload.readmitted = filters.value.readmitted
  if (filters.value.race.length) payload.race = filters.value.race
  if (filters.value.medical_specialty) payload.medical_specialty = `%${filters.value.medical_specialty}%`
  if (minRisk.value > 0) payload.risk_score = `>=${minRisk.value}`
  return payload
}

async function loadPatients() {
  loading.value = true
  try {
    const payload = buildFilterPayload()
    const { data } = await searchPatients({
      page: page.value,
      limit: limit.value,
      sortBy: 'risk_score',
      order: 'desc',
      filters: JSON.stringify(payload),
    })
    patients.value = data.data
    total.value = data.pagination.total
  } catch (err) {
    console.error(err)
  } finally {
    loading.value = false
  }
}

function onPageChange(e) {
  page.value = e.page + 1
  loadPatients()
}

function onRowsChange(rows) {
  limit.value = rows
  page.value = 1
  loadPatients()
}

function clearFilters() {
  filters.value = { age: [], gender: [], risk_class: [], readmitted: [], race: [], medical_specialty: '' }
  minRisk.value = 0
  page.value = 1
  loadPatients()
}

async function onRowSelect(e) {
  try {
    const base = e.data
    const { data } = await predictPatient(base.encounter_id)
    selectedPatient.value = {
      ...base,
      ...data.top_features,
      predicted_class: data.predicted_class ?? base.predicted_class,
      readmitted: data.actual_class ?? base.readmitted,
      risk_score: data.risk_score ?? base.risk_score,
      risk_class: data.risk_class ?? base.risk_class,
      risk_label: data.risk_label ?? base.risk_label,
      prob_no: data.probabilities?.no ?? base.prob_no,
      prob_lt30: data.probabilities?.lt30 ?? base.prob_lt30,
      prob_gt30: data.probabilities?.gt30 ?? base.prob_gt30,
    }
  } catch (err) {
    console.error(err)
  }
}

function riskSeverity(riskClass) {
  if (riskClass === 'high') return 'danger'
  if (riskClass === 'medium') return 'warn'
  return 'success'
}

function riskBarClass(riskClass) {
  if (riskClass === 'high') return 'p-progressbar-danger'
  if (riskClass === 'medium') return 'p-progressbar-warning'
  return 'p-progressbar-success'
}

function readmittedSeverity(readmitted) {
  if (readmitted === '<30') return 'danger'
  if (readmitted === '>30') return 'warn'
  return 'success'
}

onMounted(async () => {
  await loadPatients()
  try {
    const { data } = await getStats()
    usingModel.value = data.modelPredictions ?? false
  } catch (err) {
    console.error(err)
  }
})
</script>
