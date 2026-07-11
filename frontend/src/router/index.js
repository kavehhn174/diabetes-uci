import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import Dataset from '../views/Dataset.vue'
import PromptQuery from '../views/PromptQuery.vue'
import Predict from '../views/Predict.vue'

const routes = [
  { path: '/', name: 'Dashboard', component: Dashboard },
  { path: '/dataset', name: 'Dataset', component: Dataset },
  { path: '/prompt', name: 'PromptQuery', component: PromptQuery },
  { path: '/predict', name: 'Predict', component: Predict },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
