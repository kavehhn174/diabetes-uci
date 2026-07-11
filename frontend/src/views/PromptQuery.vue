<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold">Prompt Query</h1>
      <p class="text-surface-500 dark:text-surface-400">
        Ask for patients in plain English. The backend turns the prompt into a structured query and returns matching records.
      </p>
    </div>

    <div class="card space-y-4">
      <div class="flex gap-2">
        <InputText
          v-model="prompt"
          placeholder="e.g. high risk female patients over 60 on insulin readmitted within 30 days"
          class="w-full"
          @keyup.enter="execute"
        />
        <Button label="Ask" icon="pi pi-send" :loading="loading" @click="execute" />
      </div>

      <div class="flex flex-wrap gap-2 text-sm">
        <span class="text-surface-500 dark:text-surface-400">Try:</span>
        <button
          v-for="example in examples"
          :key="example"
          @click="prompt = example; execute()"
          class="rounded-full border border-surface-200 bg-surface-100 px-3 py-1 transition hover:bg-primary-100 hover:text-primary-700 dark:border-surface-700 dark:bg-surface-800 dark:hover:bg-primary-900 dark:hover:text-primary-300"
        >
          {{ example }}
        </button>
      </div>

      <div
        v-if="llmError"
        class="rounded-lg border border-amber-300 bg-amber-50 p-4 text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200"
      >
        <p class="text-sm">
          <i class="pi pi-exclamation-triangle mr-1"></i>
          <strong>LLM call failed, using rule-based fallback:</strong> {{ llmError }}
        </p>
      </div>

      <div v-if="explanation" class="rounded-lg border border-surface-200 bg-surface-50 p-4 dark:border-surface-700 dark:bg-surface-800">
        <p class="text-sm">
          <i class="pi pi-sparkles mr-1 text-primary-500"></i>
          <strong>Interpreted query</strong>
          <Tag
            :value="llmUsed ? 'LLM' : 'Rule-based fallback'"
            :severity="llmUsed ? 'info' : 'secondary'"
            class="ml-1"
          />
          : {{ explanation }}
        </p>
        <pre class="mt-2 overflow-auto rounded bg-surface-100 p-2 text-xs dark:bg-surface-900">{{ JSON.stringify(query, null, 2) }}</pre>
      </div>

      <div
        v-if="promptPrediction"
        class="rounded-lg border border-primary-200 bg-primary-50 p-4 dark:border-primary-800 dark:bg-primary-950/40"
      >
        <div class="mb-3 flex items-center justify-between gap-3">
          <h3 class="text-base font-bold">Predicted readmission for this prompt</h3>
          <Tag
            :value="`Class: ${promptPrediction.predictedClass}`"
            :severity="promptPrediction.predictedClass === 'NO' ? 'success' : promptPrediction.predictedClass === '<30' ? 'danger' : 'warn'"
          />
        </div>
        <div class="grid gap-3 sm:grid-cols-2">
          <div class="rounded-lg bg-white/80 p-3 dark:bg-surface-900/50">
            <div class="mb-1 text-xs text-surface-500 dark:text-surface-400">Readmission probability</div>
            <div class="text-2xl font-bold">{{ (promptPrediction.readmissionProbability * 100).toFixed(1) }}%</div>
            <div class="mt-2 text-xs text-surface-500 dark:text-surface-400">
              Cohort size: {{ promptPrediction.cohortSize.toLocaleString() }}
            </div>
          </div>
          <div class="space-y-2 rounded-lg bg-white/80 p-3 dark:bg-surface-900/50">
            <div class="text-xs text-surface-500 dark:text-surface-400">Class probabilities</div>
            <div class="flex items-center gap-2">
              <span class="w-16 text-xs">NO</span>
              <ProgressBar :value="promptPrediction.probabilities.no * 100" :showValue="false" class="h-2 w-full" />
              <span class="w-12 text-right text-xs">{{ (promptPrediction.probabilities.no * 100).toFixed(0) }}%</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="w-16 text-xs">&lt;30</span>
              <ProgressBar :value="promptPrediction.probabilities.lt30 * 100" :showValue="false" class="h-2 w-full" />
              <span class="w-12 text-right text-xs">{{ (promptPrediction.probabilities.lt30 * 100).toFixed(0) }}%</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="w-16 text-xs">&gt;30</span>
              <ProgressBar :value="promptPrediction.probabilities.gt30 * 100" :showValue="false" class="h-2 w-full" />
              <span class="w-12 text-right text-xs">{{ (promptPrediction.probabilities.gt30 * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
        <p
          v-if="promptPrediction.note"
          class="mt-3 text-xs text-surface-600 dark:text-surface-300"
        >
          <i class="pi pi-info-circle mr-1"></i>
          {{ promptPrediction.note }}
        </p>
      </div>
    </div>

    <div v-if="results.length" class="card">
      <div class="mb-4 flex items-center justify-between">
        <h2 class="text-lg font-bold">Results ({{ total.toLocaleString() }})</h2>
        <div class="flex items-center gap-2">
          <Button
            icon="pi pi-chevron-left"
            text
            :disabled="page === 1"
            @click="page--; execute(false)"
          />
          <span class="text-sm">Page {{ page }} of {{ pages }}</span>
          <Button
            icon="pi pi-chevron-right"
            text
            :disabled="page >= pages"
            @click="page++; execute(false)"
          />
        </div>
      </div>
      <DataTable :value="results" class="p-datatable-sm" scrollable scrollHeight="400px">
        <Column field="encounter_id" header="Encounter ID" />
        <Column field="age" header="Age" />
        <Column field="gender" header="Gender" />
        <Column field="race" header="Race" />
        <Column field="risk_score" header="Risk">
          <template #body="{ data }">
            <div class="flex items-center gap-2">
              <ProgressBar :value="data.risk_score * 100" :class="riskBarClass(data.risk_class)" class="w-20" />
              <span class="text-sm">{{ (data.risk_score * 100).toFixed(0) }}%</span>
            </div>
          </template>
        </Column>
        <Column field="risk_label" header="Class">
          <template #body="{ data }">
            <Tag :value="data.risk_label" :severity="riskSeverity(data.risk_class)" />
          </template>
        </Column>
        <Column field="readmitted" header="Actual">
          <template #body="{ data }">
            <Tag :value="data.readmitted" :severity="readmittedSeverity(data.readmitted)" />
          </template>
        </Column>
      </DataTable>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { executePrompt } from '../services/api'

import Button from 'primevue/button'
import Column from 'primevue/column'
import DataTable from 'primevue/datatable'
import InputText from 'primevue/inputtext'
import ProgressBar from 'primevue/progressbar'
import Tag from 'primevue/tag'

const prompt = ref('')
const loading = ref(false)
const results = ref([])
const total = ref(0)
const page = ref(1)
const pages = ref(0)
const explanation = ref('')
const query = ref({})
const promptPrediction = ref(null)
const llmError = ref('')
const llmUsed = ref(false)

const examples = [
  'high risk patients over 70',
  'female patients on insulin with moderate risk',
  'patients readmitted within 30 days',
  'Caucasian patients with emergency visits >= 2',
]

async function execute(resetPage = true) {
  if (!prompt.value.trim()) return
  if (resetPage) page.value = 1
  loading.value = true
  try {
    const { data } = await executePrompt(prompt.value, page.value, 20)
    results.value = data.data
    total.value = data.pagination.total
    pages.value = data.pagination.pages
    explanation.value = data.query.explanation
    query.value = data.query
    promptPrediction.value = data.promptPrediction
    llmError.value = data.query.llmError || ''
    llmUsed.value = !!data.query.llmUsed
  } catch (err) {
    console.error(err)
    llmError.value = err?.response?.data?.error || err.message || 'Request failed'
    llmUsed.value = false
  } finally {
    loading.value = false
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
</script>
