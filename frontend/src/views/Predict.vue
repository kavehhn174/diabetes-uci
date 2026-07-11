<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold">Predict a Query</h1>
      <p class="text-surface-500 dark:text-surface-400">
        Enter a custom patient profile and run it through the trained Bayesian Network
        (<code>cleaned_hill_climb_bn_model.pkl</code>) directly. This performs real exact inference
        (variable elimination) over the learned conditional probability tables — not a lookup of a
        precomputed value.
      </p>
      <div class="mt-2 flex items-center gap-2 text-sm">
        <span
          class="inline-block h-2 w-2 rounded-full"
          :class="serviceHealthy ? 'bg-emerald-500' : 'bg-red-500'"
        ></span>
        <span class="text-surface-500 dark:text-surface-400">
          Model service: {{ serviceHealthy ? 'online' : 'starting / unavailable' }}
        </span>
      </div>
    </div>

    <div class="card grid items-start gap-6 lg:grid-cols-2">
      <div class="space-y-4">
        <h2 class="text-lg font-bold">Patient features</h2>

        <div class="grid gap-4 sm:grid-cols-2">
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Age group</label>
            <Select v-model="form.age_group" :options="ageGroupOptions" class="w-full" />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Admission type</label>
            <Select v-model="form.admission_type" :options="admissionTypeOptions" class="w-full" />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Admission source</label>
            <Select v-model="form.admission_source" :options="admissionSourceOptions" class="w-full" />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Discharge disposition</label>
            <Select v-model="form.discharge_disposition" :options="dischargeDispositionOptions" class="w-full" />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Primary diagnosis category</label>
            <Select v-model="form.diag_1_group" :options="diagGroupOptions" class="w-full" />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Medical specialty</label>
            <InputText v-model="form.medical_specialty" placeholder="e.g. Cardiology (blank = Missing)" class="w-full" />
          </div>
        </div>

        <h3 class="pt-2 text-sm font-bold text-surface-500 dark:text-surface-400">Utilization counts</h3>
        <div class="grid gap-4 sm:grid-cols-3">
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Prior inpatient visits</label>
            <InputNumber v-model="form.number_inpatient" :min="0" showButtons fluid />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Prior outpatient visits</label>
            <InputNumber v-model="form.number_outpatient" :min="0" showButtons fluid />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Prior emergency visits</label>
            <InputNumber v-model="form.number_emergency" :min="0" showButtons fluid />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Number of diagnoses</label>
            <InputNumber v-model="form.number_diagnoses" :min="0" showButtons fluid />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Time in hospital (days)</label>
            <InputNumber v-model="form.time_in_hospital" :min="0" showButtons fluid />
          </div>
          <div class="flex min-w-0 flex-col gap-1">
            <label class="text-sm font-medium">Number of medications</label>
            <InputNumber v-model="form.num_medications" :min="0" showButtons fluid />
          </div>
        </div>

        <div class="flex flex-wrap gap-2 pt-2">
          <Button label="Predict" icon="pi pi-bolt" :loading="loading" @click="runPredict" />
          <Button label="High-risk example" text @click="loadExample('high')" />
          <Button label="Low-risk example" text @click="loadExample('low')" />
        </div>

        <p v-if="error" class="text-sm text-red-500">{{ error }}</p>
      </div>

      <div>
        <h2 class="mb-2 text-lg font-bold">Prediction</h2>
        <div v-if="!result" class="flex min-h-[220px] items-center justify-center rounded-lg border border-dashed border-surface-300 text-sm text-surface-400 dark:border-surface-600">
          Run a prediction to see the model's output here.
        </div>
        <div v-else class="space-y-4 rounded-lg border border-primary-200 bg-primary-50 p-4 dark:border-primary-800 dark:bg-primary-950/40">
          <div class="flex items-center justify-between gap-3">
            <h3 class="text-base font-bold">Predicted class</h3>
            <Tag
              :value="result.predicted_class"
              :severity="result.predicted_class === 'NO' ? 'success' : result.predicted_class === '<30' ? 'danger' : 'warn'"
            />
          </div>

          <div class="rounded-lg bg-white/80 p-3 dark:bg-surface-900/50">
            <div class="mb-1 text-xs text-surface-500 dark:text-surface-400">Risk score</div>
            <div class="text-2xl font-bold">{{ (result.risk_score * 100).toFixed(1) }}%</div>
            <Tag class="mt-2" :value="result.risk_label" :severity="riskSeverity(result.risk_class)" />
          </div>

          <div class="space-y-2 rounded-lg bg-white/80 p-3 dark:bg-surface-900/50">
            <div class="text-xs text-surface-500 dark:text-surface-400">Class probabilities</div>
            <div v-for="cls in ['NO', '<30', '>30']" :key="cls" class="flex items-center gap-2">
              <span class="w-12 text-xs">{{ cls }}</span>
              <ProgressBar :value="(result.probabilities[cls] || 0) * 100" :showValue="false" class="h-2 w-full" />
              <span class="w-12 text-right text-xs">{{ ((result.probabilities[cls] || 0) * 100).toFixed(1) }}%</span>
            </div>
          </div>

          <div class="rounded-lg bg-white/80 p-3 text-xs dark:bg-surface-900/50">
            <div class="mb-1 text-surface-500 dark:text-surface-400">Cleaned features sent to the model</div>
            <pre class="overflow-auto text-xs">{{ JSON.stringify(result.input_features, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted } from 'vue'
import { predictQuery, getPredictHealth } from '../services/api'

import Button from 'primevue/button'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import ProgressBar from 'primevue/progressbar'
import Select from 'primevue/select'
import Tag from 'primevue/tag'

const ageGroupOptions = ['<30', '30-60', '>60']
const admissionTypeOptions = ['Emergency', 'Urgent', 'Elective', 'Newborn', 'Trauma Center', 'Not Available']
const admissionSourceOptions = [
  'Emergency room', 'Physician referral', 'Clinic referral', 'Transfer from hospital',
  'Transfer from other health facility', 'Not Available', 'Other',
]
const dischargeDispositionOptions = [
  'Discharged to home', 'Transferred to SNF', 'Home with home health service',
  'Transferred to another hospital', 'Transferred to rehab facility', 'Not Available', 'Other',
]
const diagGroupOptions = ['Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury', 'Other', 'Missing']

const defaultForm = {
  age_group: '>60',
  admission_type: 'Emergency',
  admission_source: 'Emergency room',
  discharge_disposition: 'Discharged to home',
  diag_1_group: 'Circulatory',
  medical_specialty: 'Cardiology',
  number_inpatient: 2,
  number_outpatient: 0,
  number_emergency: 1,
  number_diagnoses: 9,
  time_in_hospital: 4,
  num_medications: 18,
}

const form = reactive({ ...defaultForm })
const result = ref(null)
const error = ref('')
const loading = ref(false)
const serviceHealthy = ref(false)

function riskSeverity(riskClass) {
  if (riskClass === 'high') return 'danger'
  if (riskClass === 'medium') return 'warn'
  return 'success'
}

function loadExample(kind) {
  if (kind === 'high') {
    Object.assign(form, defaultForm)
  } else {
    Object.assign(form, {
      age_group: '<30',
      admission_type: 'Elective',
      admission_source: 'Physician referral',
      discharge_disposition: 'Discharged to home',
      diag_1_group: 'Other',
      medical_specialty: 'Family/GeneralPractice',
      number_inpatient: 0,
      number_outpatient: 0,
      number_emergency: 0,
      number_diagnoses: 3,
      time_in_hospital: 1,
      num_medications: 4,
    })
  }
  result.value = null
  error.value = ''
}

async function runPredict() {
  loading.value = true
  error.value = ''
  try {
    const { data } = await predictQuery({ ...form })
    result.value = data
    serviceHealthy.value = true
  } catch (err) {
    error.value = err.response?.data?.error || 'Prediction failed. Is the model service running?'
    console.error(err)
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    const { data } = await getPredictHealth()
    serviceHealthy.value = !!data.model_loaded
  } catch (err) {
    serviceHealthy.value = false
  }
})
</script>
