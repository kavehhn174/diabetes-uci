<template>
  <div :class="{ dark: isDark }" class="min-h-screen bg-surface-50 text-surface-900 transition-colors dark:bg-surface-900 dark:text-surface-0">
    <nav class="sticky top-0 z-50 border-b border-surface-200 bg-white/80 backdrop-blur dark:border-surface-700 dark:bg-surface-800/80">
      <div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        <div class="flex items-center gap-2">
          <i class="pi pi-chart-line text-primary-500 text-2xl"></i>
          <span class="text-lg font-bold">Readmission Risk</span>
        </div>
        <div class="flex items-center gap-1 sm:gap-4">
          <router-link
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            class="rounded-lg px-3 py-2 text-sm font-medium transition-colors hover:bg-surface-100 dark:hover:bg-surface-700"
            :class="{ 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300': $route.path === item.path }"
          >
            <i :class="item.icon" class="mr-1"></i>
            <span class="hidden sm:inline">{{ item.label }}</span>
          </router-link>
          <button
            @click="toggleDark"
            class="ml-2 rounded-lg p-2 transition-colors hover:bg-surface-100 dark:hover:bg-surface-700"
            :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
          >
            <i :class="isDark ? 'pi pi-sun' : 'pi pi-moon'"></i>
          </button>
        </div>
      </div>
    </nav>
    <main class="mx-auto max-w-7xl p-4 sm:p-6">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const isDark = ref(localStorage.getItem('theme') === 'dark')
const route = useRoute()

const navItems = [
  { path: '/', label: 'Patients', icon: 'pi pi-users' },
  { path: '/prompt', label: 'Prompt Query', icon: 'pi pi-comment' },
  { path: '/predict', label: 'Predict', icon: 'pi pi-bolt' },
  { path: '/dataset', label: 'Dataset', icon: 'pi pi-database' },
]

function toggleDark() {
  isDark.value = !isDark.value
  localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
  if (isDark.value) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

if (isDark.value) {
  document.documentElement.classList.add('dark')
}

watch(route, () => {
  // Any route change logic can go here
})
</script>
