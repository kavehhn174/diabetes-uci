import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// The LLM-backed prompt endpoints can take longer than typical API calls
// (the hosted reasoning model can take 15-20s+ to respond) before the
// backend's own timeout kicks in and falls back to the rule-based parser.
// Give these a longer client-side timeout so we don't bail out before the
// backend has a chance to respond with either result.
const LLM_TIMEOUT_MS = 35000

export const health = () => api.get('/health')

export const searchPatients = (params) => api.get('/patients/search', { params })

export const getPatient = (id) => api.get(`/patients/${id}`)

export const predictPatient = (encounter_id) => api.post('/patients/predict', { encounter_id })

export const getStats = () => api.get('/stats/summary')

export const getFeatureStats = (feature) => api.get(`/stats/readmitted-by-feature/${feature}`)

export const nlpQuery = (prompt) => api.post('/query/nlp', { prompt }, { timeout: LLM_TIMEOUT_MS })

export const executePrompt = (prompt, page = 1, limit = 20) =>
  api.post('/query/execute', { prompt, page, limit }, { timeout: LLM_TIMEOUT_MS })

// Runs a live forward pass through the trained Bayesian Network
// (cleaned_hill_climb_bn_model.pkl) for a single custom patient query.
export const predictQuery = (query) => api.post('/predict', { query })

export const getPredictHealth = () => api.get('/predict/health')

export default api
