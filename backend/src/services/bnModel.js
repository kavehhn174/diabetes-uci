/**
 * Manages the Python microservice (backend/model/server.py) that runs live
 * inference through the trained Bayesian Network stored in
 * backend/model/cleaned_hill_climb_bn_model.pkl, and exposes a `predict()`
 * helper that Express routes can call.
 *
 * The service is started lazily (on first prediction request) and reused
 * for subsequent requests. If it's already running (e.g. started manually
 * with `./venv/bin/python server.py`), we just talk to it over HTTP.
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const MODEL_DIR = path.join(__dirname, '..', '..', 'model');
const SERVER_SCRIPT = path.join(MODEL_DIR, 'server.py');
const SERVICE_URL = process.env.BN_MODEL_URL || 'http://127.0.0.1:5001';
const START_TIMEOUT_MS = 30000;
const HEALTH_TIMEOUT_MS = 1500;

let pythonProcess = null;
let startingPromise = null;

function getPythonCandidates() {
  const candidates = [];
  if (process.env.PYTHON_BIN) candidates.push(process.env.PYTHON_BIN);
  const venvPython = path.join(MODEL_DIR, 'venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) candidates.push(venvPython);
  candidates.push('python3', 'python');
  return candidates;
}

async function checkHealth(timeoutMs = HEALTH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${SERVICE_URL}/health`, { signal: controller.signal });
    if (!res.ok) return null;
    const body = await res.json();
    return body.model_loaded ? body : null;
  } catch (_err) {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function spawnPythonService() {
  const candidates = getPythonCandidates();
  const python = candidates[0];

  const child = spawn(python, [SERVER_SCRIPT], {
    cwd: MODEL_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  });

  child.stdout.on('data', (chunk) => {
    process.stdout.write(`[bn-model] ${chunk}`);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(`[bn-model] ${chunk}`);
  });
  child.on('exit', (code, signal) => {
    console.warn(`[bn-model] Python prediction service exited (code=${code}, signal=${signal})`);
    if (pythonProcess === child) pythonProcess = null;
  });
  child.on('error', (err) => {
    console.error('[bn-model] Failed to start Python prediction service:', err.message);
    if (pythonProcess === child) pythonProcess = null;
  });

  return child;
}

async function ensureRunning() {
  const alreadyHealthy = await checkHealth();
  if (alreadyHealthy) return alreadyHealthy;

  if (startingPromise) return startingPromise;

  startingPromise = (async () => {
    if (!pythonProcess) {
      console.log('[bn-model] Starting Python Bayesian Network prediction service...');
      pythonProcess = spawnPythonService();
    }

    const deadline = Date.now() + START_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const health = await checkHealth();
      if (health) return health;
      await sleep(400);
    }
    throw new Error(
      `Python prediction service did not become healthy within ${START_TIMEOUT_MS}ms. ` +
      'Check that backend/model/venv has flask/pgmpy/joblib installed, or start it manually with ' +
      '`./model/venv/bin/python model/server.py`.'
    );
  })();

  try {
    return await startingPromise;
  } finally {
    startingPromise = null;
  }
}

class BNModelError extends Error {
  constructor(message, statusCode = 500, details = null) {
    super(message);
    this.statusCode = statusCode;
    this.details = details;
  }
}

async function predict(query) {
  await ensureRunning();

  let response;
  try {
    response = await fetch(`${SERVICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
  } catch (err) {
    throw new BNModelError(`Could not reach the Bayesian Network prediction service: ${err.message}`, 503);
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new BNModelError(body.error || 'Prediction failed', response.status, body.missing);
  }
  return body;
}

async function getHealth() {
  const health = await checkHealth(3000);
  if (health) return health;
  return { status: 'unavailable', model_loaded: false };
}

module.exports = { predict, getHealth, ensureRunning, BNModelError };
