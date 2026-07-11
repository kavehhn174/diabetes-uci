require('dotenv').config({ quiet: true });

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');

const { getDb, initSchema } = require('./src/database');
const patientsRouter = require('./src/routes/patients');
const statsRouter = require('./src/routes/stats');
const queryRouter = require('./src/routes/query');
const predictRouter = require('./src/routes/predict');
const { ensureRunning: ensureBnModelRunning } = require('./src/services/bnModel');

const app = express();
const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(morgan(NODE_ENV === 'production' ? 'combined' : 'dev'));

const limiter = rateLimit({
  windowMs: 1 * 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// Initialise DB schema on startup
const db = getDb();
initSchema(db);
db.close();

app.use('/api/patients', patientsRouter);
app.use('/api/stats', statsRouter);
app.use('/api/query', queryRouter);
app.use('/api/predict', predictRouter);

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

if (NODE_ENV === 'production') {
  const distPath = path.join(__dirname, '..', 'frontend', 'dist');
  app.use(express.static(distPath));
  app.get('*', (_req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  // Warm up the Bayesian Network prediction microservice in the background
  // so the first real /api/predict request doesn't pay the model-load cost.
  ensureBnModelRunning().catch((err) => {
    console.warn(`Bayesian Network prediction service not ready yet: ${err.message}`);
  });
});
