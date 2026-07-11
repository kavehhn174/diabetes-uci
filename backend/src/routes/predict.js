const express = require('express');
const { predict, getHealth, BNModelError } = require('../services/bnModel');

const router = express.Router();

// GET /api/predict/health -- whether the live model microservice is up.
router.get('/health', async (_req, res) => {
  const health = await getHealth();
  res.json(health);
});

// POST /api/predict
// Body: { query: { ...patient features... } }
// Runs a real forward pass through the trained Bayesian Network
// (cleaned_hill_climb_bn_model.pkl) via exact inference and returns the
// predicted probability distribution over `readmitted`.
router.post('/', async (req, res) => {
  const query = req.body && typeof req.body.query === 'object' ? req.body.query : req.body;
  if (!query || typeof query !== 'object' || Array.isArray(query) || !Object.keys(query).length) {
    return res.status(400).json({ error: 'Request body must include a non-empty "query" object of patient features' });
  }

  try {
    const result = await predict(query);
    res.json(result);
  } catch (err) {
    if (err instanceof BNModelError) {
      return res.status(err.statusCode).json({ error: err.message, missing: err.details || undefined });
    }
    console.error('Prediction failed:', err);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

module.exports = router;
