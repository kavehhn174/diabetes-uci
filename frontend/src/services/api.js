import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export const health = () => api.get('/health')

export const searchPatients = (params) => api.get('/patients/search', { params })

export const getPatient = (id) => api.get(`/patients/${id}`)

export const predictPatient = (encounter_id) => api.post('/patients/predict', { encounter_id })

export const getStats = () => api.get('/stats/summary')

export const getFeatureStats = (feature) => api.get(`/stats/readmitted-by-feature/${feature}`)

export const nlpQuery = (prompt) => api.post('/query/nlp', { prompt })

export const executePrompt = (prompt, page = 1, limit = 20) =>
  api.post('/query/execute', { prompt, page, limit })

// Runs a live forward pass through the trained Bayesian Network
// (cleaned_hill_climb_bn_model.pkl) for a single custom patient query.
export const predictQuery = (query) => api.post('/predict', { query })

export const getPredictHealth = () => api.get('/predict/health')

export default api
