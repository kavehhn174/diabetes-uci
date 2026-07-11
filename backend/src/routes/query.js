const express = require('express');
const { getDb } = require('../database');

const router = express.Router();

const AGE_ORDER = [
  '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
  '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
];

const RACE_VALUES = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'];
const GENDER_VALUES = ['Male', 'Female', 'Unknown/Invalid'];
const NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1';
const NIM_MODEL = 'mistralai/ministral-14b-instruct-2512';
// Optional: set NIM_API_KEY in the environment (see .env.example) to enable
// LLM-backed prompt parsing. Without it, the rule-based fallback parser
// below (parseNaturalLanguage) is used automatically.
const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ALLOWED_FILTER_KEYS = new Set([
  'age',
  'gender',
  'race',
  'risk_class',
  'readmitted',
  'medical_specialty',
  'number_inpatient',
  'number_emergency',
  'number_diagnoses',
  'time_in_hospital',
  'diabetesMed',
  'insulin',
]);

const ALLOWED_SORT_BY = new Set([
  'risk_score',
  'encounter_id',
  'time_in_hospital',
  'number_inpatient',
  'number_emergency',
  'number_diagnoses',
  'age',
]);

function ageFromNumber(n) {
  if (n < 10) return ['[0-10)'];
  if (n < 20) return ['[10-20)'];
  if (n < 30) return ['[20-30)'];
  if (n < 40) return ['[30-40)'];
  if (n < 50) return ['[40-50)'];
  if (n < 60) return ['[50-60)'];
  if (n < 70) return ['[60-70)'];
  if (n < 80) return ['[70-80)'];
  if (n < 90) return ['[80-90)'];
  return ['[90-100)'];
}

function agesInRange(min, max) {
  return AGE_ORDER.filter((a) => {
    const start = parseInt(a.match(/\[(\d+)-/)?.[1] ?? 0, 10);
    const end = parseInt(a.match(/-(\d+)\)/)?.[1] ?? 100, 10);
    return start >= min && end <= max;
  });
}

function parseNaturalLanguage(prompt) {
  const lower = prompt.toLowerCase();
  const filters = {};
  let sortBy = 'risk_score';
  let order = 'desc';
  let explanationParts = [];

  // Age
  const ageRangeMatch = lower.match(/(?:age|patients)?\s*(\d+)\s*(?:and|to)\s*(\d+)\s*(?:years?\s*old)?|age\s*(?:between)?\s*(\d+)\s*(?:and|to)?\s*(\d+)/);
  if (ageRangeMatch) {
    const min = Number(ageRangeMatch[1] || ageRangeMatch[3]);
    const max = Number(ageRangeMatch[2] || ageRangeMatch[4]);
    if (min && max) {
      filters.age = agesInRange(min, max);
      explanationParts.push(`age between ${min} and ${max}`);
    }
  } else {
    const ageOver = lower.match(/(?:older?\s*than|age\s*over|age\s*>|above\s*age|over\s*(?:age\s*)?|patients\s*over)\s*(\d+)(?:\s*years?\s*old)?/);
    const ageUnder = lower.match(/(?:younger?\s*than|age\s*under|age\s*<|below\s*age|under\s*(?:age\s*)?|patients\s*under)\s*(\d+)(?:\s*years?\s*old)?/);
    if (ageOver) {
      const n = Number(ageOver[1]);
      filters.age = AGE_ORDER.filter((a) => {
        const start = parseInt(a.match(/\[(\d+)-/)?.[1] ?? 0, 10);
        return start >= n;
      });
      explanationParts.push(`age over ${n}`);
    } else if (ageUnder) {
      const n = Number(ageUnder[1]);
      filters.age = AGE_ORDER.filter((a) => {
        const end = parseInt(a.match(/-(\d+)\)/)?.[1] ?? 100, 10);
        return end <= n;
      });
      explanationParts.push(`age under ${n}`);
    }
  }

  // Gender
  if (lower.includes('male') && !lower.includes('female')) {
    filters.gender = ['Male'];
    explanationParts.push('male patients');
  } else if (lower.includes('female')) {
    filters.gender = ['Female'];
    explanationParts.push('female patients');
  }

  // Race
  for (const race of RACE_VALUES) {
    if (lower.includes(race.toLowerCase())) {
      filters.race = [race];
      explanationParts.push(`${race} patients`);
      break;
    }
  }

  // Risk class
  if (lower.includes('high risk')) {
    filters.risk_class = ['high'];
    explanationParts.push('high readmission risk');
  } else if (lower.includes('low risk')) {
    filters.risk_class = ['low'];
    explanationParts.push('low readmission risk');
  } else if (lower.includes('moderate risk') || lower.includes('medium risk')) {
    filters.risk_class = ['medium'];
    explanationParts.push('moderate readmission risk');
  }

  // Readmitted status
  if (lower.includes('readmitted') || lower.includes('readmission')) {
    if (lower.includes('not readmitted') || lower.includes('no readmission')) {
      filters.readmitted = ['NO'];
      explanationParts.push('not readmitted');
    } else if (lower.includes('within 30') || lower.includes('<30')) {
      filters.readmitted = ['<30'];
      explanationParts.push('readmitted within 30 days');
    } else if (lower.includes('after 30') || lower.includes('>30')) {
      filters.readmitted = ['>30'];
      explanationParts.push('readmitted after 30 days');
    } else {
      filters.readmitted = ['<30', '>30'];
      explanationParts.push('readmitted at any time');
    }
  }

  // Medical specialty
  const specialtyMatch = lower.match(/(?:specialty|specialist|doctor)\s*(?:is)?\s*([a-z\s]+)/);
  if (specialtyMatch) {
    const candidate = specialtyMatch[1].trim();
    if (candidate.length > 2) {
      filters.medical_specialty = `%${candidate}%`;
      explanationParts.push(`medical specialty matching "${candidate}"`);
    }
  }

  // Numeric filters
  const numberInpatientMatch = lower.match(/(?:inpatient|hospital admissions?)\s*(>=|<=|>|<|=)?\s*(\d+)/);
  if (numberInpatientMatch) {
    const op = numberInpatientMatch[1] || '>=';
    const val = Number(numberInpatientMatch[2]);
    filters.number_inpatient = `${op}${val}`;
    explanationParts.push(`inpatient visits ${op} ${val}`);
  }

  const emergencyMatch = lower.match(/(?:emergency|er visits?)\s*(>=|<=|>|<|=)?\s*(\d+)/);
  if (emergencyMatch) {
    const op = emergencyMatch[1] || '>=';
    const val = Number(emergencyMatch[2]);
    filters.number_emergency = `${op}${val}`;
    explanationParts.push(`emergency visits ${op} ${val}`);
  }

  const diagnosesMatch = lower.match(/(?:diagnoses?)\s*(>=|<=|>|<|=)?\s*(\d+)/);
  if (diagnosesMatch) {
    const op = diagnosesMatch[1] || '>=';
    const val = Number(diagnosesMatch[2]);
    filters.number_diagnoses = `${op}${val}`;
    explanationParts.push(`diagnoses ${op} ${val}`);
  }

  const timeInHospitalMatch = lower.match(/(?:time in hospital|hospital stay|days in hospital)\s*(>=|<=|>|<|=)?\s*(\d+)/);
  if (timeInHospitalMatch) {
    const op = timeInHospitalMatch[1] || '>=';
    const val = Number(timeInHospitalMatch[2]);
    filters.time_in_hospital = `${op}${val}`;
    explanationParts.push(`time in hospital ${op} ${val} days`);
  }

  // Medication / diabetesMed
  if (lower.includes('on diabetes medication') || lower.includes('diabetes med yes')) {
    filters.diabetesMed = ['Yes'];
    explanationParts.push('on diabetes medication');
  } else if (lower.includes('no diabetes medication') || lower.includes('diabetes med no')) {
    filters.diabetesMed = ['No'];
    explanationParts.push('not on diabetes medication');
  }

  if (lower.includes('insulin')) {
    if (lower.includes('no insulin')) {
      filters.insulin = ['No'];
      explanationParts.push('not on insulin');
    } else {
      filters.insulin = ['Up', 'Down', 'Steady'];
      explanationParts.push('on insulin');
    }
  }

  // Sorting
  if (lower.includes('newest') || lower.includes('latest')) {
    sortBy = 'encounter_id';
    order = 'desc';
  } else if (lower.includes('oldest')) {
    sortBy = 'encounter_id';
    order = 'asc';
  } else if (lower.includes('highest risk') || lower.includes('most at risk')) {
    sortBy = 'risk_score';
    order = 'desc';
  }

  const explanation = explanationParts.length
    ? `Querying patients with ${explanationParts.join(', ')}.`
    : 'Showing all patients sorted by risk score.';

  return { filters, sortBy, order, explanation };
}

function extractJsonObject(text) {
  if (!text || typeof text !== 'string') return null;
  const fenced = text.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    try {
      return JSON.parse(fenced[1].trim());
    } catch (_err) {
      // fall through
    }
  }
  const first = text.indexOf('{');
  const last = text.lastIndexOf('}');
  if (first >= 0 && last > first) {
    try {
      return JSON.parse(text.slice(first, last + 1));
    } catch (_err) {
      return null;
    }
  }
  return null;
}

function normalizeLLMQuery(result, prompt) {
  const fallback = parseNaturalLanguage(prompt);
  if (!result || typeof result !== 'object') return fallback;

  const filters = {};
  const incomingFilters = result.filters && typeof result.filters === 'object' ? result.filters : {};
  for (const [key, value] of Object.entries(incomingFilters)) {
    if (!ALLOWED_FILTER_KEYS.has(key)) continue;
    if (value === null || value === undefined || value === '') continue;
    filters[key] = value;
  }

  // Normalize common LLM outputs to actual dataset values.
  if (Array.isArray(filters.insulin)) {
    const normalized = filters.insulin.map((v) => String(v).toLowerCase());
    if (normalized.includes('yes') || normalized.includes('true')) {
      filters.insulin = ['Up', 'Down', 'Steady'];
    } else if (normalized.includes('no') || normalized.includes('false')) {
      filters.insulin = ['No'];
    }
  } else if (typeof filters.insulin === 'string') {
    const insulinValue = filters.insulin.toLowerCase();
    if (insulinValue === 'yes' || insulinValue === 'true') {
      filters.insulin = ['Up', 'Down', 'Steady'];
    } else if (insulinValue === 'no' || insulinValue === 'false') {
      filters.insulin = ['No'];
    }
  }

  if (Array.isArray(filters.diabetesMed)) {
    filters.diabetesMed = filters.diabetesMed.map((v) => {
      const lower = String(v).toLowerCase();
      if (lower === 'true') return 'Yes';
      if (lower === 'false') return 'No';
      return lower === 'yes' ? 'Yes' : lower === 'no' ? 'No' : v;
    });
  }

  if (Array.isArray(filters.gender)) {
    const mapGender = { male: 'Male', female: 'Female', 'unknown/invalid': 'Unknown/Invalid' };
    filters.gender = filters.gender.map((v) => mapGender[String(v).toLowerCase()] || v);
  }

  if (Array.isArray(filters.race)) {
    const mapRace = {
      caucasian: 'Caucasian',
      africanamerican: 'AfricanAmerican',
      hispanic: 'Hispanic',
      asian: 'Asian',
      other: 'Other',
      unknown: 'Unknown',
    };
    filters.race = filters.race.map((v) => mapRace[String(v).toLowerCase()] || v);
  }

  if (Array.isArray(filters.risk_class)) {
    filters.risk_class = filters.risk_class.map((v) => String(v).toLowerCase());
  }

  if (Array.isArray(filters.readmitted)) {
    const mapReadmitted = { no: 'NO', '<30': '<30', '>30': '>30' };
    filters.readmitted = filters.readmitted.map((v) => mapReadmitted[String(v).toLowerCase()] || v);
  }

  // Numeric comparison filters must be scalar operator strings (e.g. ">=3"),
  // but the LLM sometimes wraps them in a single-element array - unwrap those.
  for (const key of ['number_inpatient', 'number_emergency', 'number_diagnoses', 'time_in_hospital']) {
    if (Array.isArray(filters[key])) {
      filters[key] = filters[key].length ? String(filters[key][0]) : undefined;
      if (filters[key] === undefined) delete filters[key];
    }
  }

  const sortBy = ALLOWED_SORT_BY.has(result.sortBy) ? result.sortBy : fallback.sortBy;
  const order = result.order === 'asc' ? 'asc' : 'desc';
  const explanation = typeof result.explanation === 'string' && result.explanation.trim()
    ? result.explanation.trim()
    : fallback.explanation;

  return { filters, sortBy, order, explanation };
}

// Reads an OpenAI-style Server-Sent-Events chat completion stream and
// accumulates the token deltas into the final text. `onChunk` is invoked on
// every chunk received (even empty keep-alives) so the caller can reset an
// idle timeout - i.e. we only give up if the model actually stalls, not
// based on the total time the full response takes to stream in.
async function readSSEStream(body, onChunk) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  let reasoningContent = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk?.();
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === '[DONE]') continue;
      try {
        const json = JSON.parse(data);
        const delta = json?.choices?.[0]?.delta || {};
        if (delta.content) content += delta.content;
        if (delta.reasoning_content) reasoningContent += delta.reasoning_content;
      } catch (_err) {
        // Ignore partial/malformed SSE lines; they get completed by the
        // next chunk or are harmless keep-alives.
      }
    }
  }

  return content || reasoningContent;
}

async function callLLM(prompt) {
  if (!NIM_API_KEY) {
    return { ...parseNaturalLanguage(prompt), llmUsed: false, llmError: null };
  }

  const systemPrompt = [
    'You convert medical search prompts into JSON for SQL filtering.',
    'Return ONLY valid JSON (no markdown).',
    'Schema:',
    '{',
    '  "filters": {',
    '    "age"?: string[],',
    '    "gender"?: string[],',
    '    "race"?: string[],',
    '    "risk_class"?: string[],',
    '    "readmitted"?: string[],',
    '    "medical_specialty"?: string,',
    '    "number_inpatient"?: string,',
    '    "number_emergency"?: string,',
    '    "number_diagnoses"?: string,',
    '    "time_in_hospital"?: string,',
    '    "diabetesMed"?: string[],',
    '    "insulin"?: string[]',
    '  },',
    '  "sortBy": "risk_score" | "encounter_id" | "time_in_hospital" | "number_inpatient" | "number_emergency" | "number_diagnoses" | "age",',
    '  "order": "asc" | "desc",',
    '  "explanation": string',
    '}',
    'Age buckets allowed: [0-10), [10-20), [20-30), [30-40), [40-50), [50-60), [60-70), [70-80), [80-90), [90-100).',
    'Risk class values allowed: low, medium, high.',
    'Readmitted values allowed: NO, <30, >30.',
    'Numeric constraints must be string ops like >=2, <5, =3.',
  ].join('\n');

  // We stream the response instead of waiting for the full completion so we
  // can use an *idle* timeout (no data received for N ms) rather than one
  // hard cutoff on total request time. That way a model that's slow but
  // steadily streaming tokens isn't killed early, while a genuinely stuck
  // request still gets cut off quickly. HARD_TIMEOUT_MS is a backstop in
  // case the connection opens but the server never streams anything at all.
  const IDLE_TIMEOUT_MS = 12000;
  const HARD_TIMEOUT_MS = 30000;
  const controller = new AbortController();
  let idleTimer;
  let timedOutReason = null;
  const armIdleTimer = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      timedOutReason = `no data received from NIM for ${IDLE_TIMEOUT_MS}ms (stream stalled)`;
      controller.abort();
    }, IDLE_TIMEOUT_MS);
  };
  const hardTimer = setTimeout(() => {
    timedOutReason = `NIM request exceeded hard timeout of ${HARD_TIMEOUT_MS}ms`;
    controller.abort();
  }, HARD_TIMEOUT_MS);

  try {
    armIdleTimer();
    const response = await fetch(`${NIM_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      signal: controller.signal,
      body: JSON.stringify({
        model: NIM_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        temperature: 0.2,
        top_p: 1,
        max_tokens: 800,
        stream: true,
      }),
    });

    if (!response.ok || !response.body) {
      const errText = await response.text().catch(() => '');
      throw new Error(`NIM API error ${response.status} ${response.statusText}: ${errText.slice(0, 500)}`);
    }

    const content = await readSSEStream(response.body, armIdleTimer);
    clearTimeout(idleTimer);
    clearTimeout(hardTimer);

    const parsed = extractJsonObject(content);
    if (!parsed) {
      const snippet = content ? content.slice(0, 300) : '(empty response)';
      throw new Error(`NIM response was not valid/parseable JSON: ${snippet}`);
    }
    return { ...normalizeLLMQuery(parsed, prompt), llmUsed: true, llmError: null };
  } catch (err) {
    clearTimeout(idleTimer);
    clearTimeout(hardTimer);
    const isAbort = err.name === 'AbortError';
    const message = isAbort ? (timedOutReason || 'NIM request was aborted') : err.message;
    console.error(`NIM LLM call failed (model=${NIM_MODEL}), using fallback parser:`, message);
    return { ...parseNaturalLanguage(prompt), llmUsed: false, llmError: message };
  }
}

router.post('/nlp', async (req, res) => {
  const { prompt } = req.body;
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'prompt is required' });
  }

  try {
    const result = await callLLM(prompt);
    res.json({ prompt, ...result });
  } catch (err) {
    console.error('LLM query failed:', err);
    res.status(500).json({ error: 'Failed to process natural language query' });
  }
});

router.post('/execute', async (req, res) => {
  const { prompt, page = 1, limit = 20 } = req.body;
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'prompt is required' });
  }

  try {
    const query = await callLLM(prompt);
    const db = getDb();

    const offset = (page - 1) * limit;
    const conditions = [];
    const params = [];

    for (const [key, val] of Object.entries(query.filters || {})) {
      if (val === undefined || val === null || val === '') continue;
      if (Array.isArray(val)) {
        conditions.push(`${key} IN (${val.map(() => '?').join(',')})`);
        params.push(...val);
      } else if (typeof val === 'string' && val.includes('%')) {
        conditions.push(`${key} LIKE ?`);
        params.push(val);
      } else if (typeof val === 'string' && /^[><=]+\d+/.test(val)) {
        const op = val.match(/^[><=]+/)[0];
        const num = Number(val.slice(op.length));
        conditions.push(`${key} ${op} ?`);
        params.push(num);
      } else {
        conditions.push(`${key} = ?`);
        params.push(val);
      }
    }

    const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
    const sortBy = /^[a-zA-Z0-9_]+$/.test(query.sortBy) ? query.sortBy : 'risk_score';
    const order = query.order === 'asc' ? 'ASC' : 'DESC';

    const countStmt = db.prepare(`SELECT COUNT(*) as total FROM patients ${where}`);
    const { total } = countStmt.get(...params);

    const buildPrediction = (count, predictionWhere, predictionParams, note = null) => {
      if (!count || count <= 0) return null;
      const aggregateStmt = db.prepare(
        `SELECT
          AVG(COALESCE(prob_no, 0)) as prob_no,
          AVG(COALESCE(prob_lt30, 0)) as prob_lt30,
          AVG(COALESCE(prob_gt30, 0)) as prob_gt30,
          AVG(COALESCE(risk_score, 0)) as risk_score
         FROM patients ${predictionWhere}`
      );
      const agg = aggregateStmt.get(...predictionParams);
      const probNo = Number(agg.prob_no || 0);
      const probLt30 = Number(agg.prob_lt30 || 0);
      const probGt30 = Number(agg.prob_gt30 || 0);
      const readmissionProbability = probLt30 + probGt30;

      const classCandidates = [
        { cls: 'NO', p: probNo },
        { cls: '<30', p: probLt30 },
        { cls: '>30', p: probGt30 },
      ].sort((a, b) => b.p - a.p);
      const predictedClass = classCandidates[0].cls;
      const confidence = classCandidates[0].p;

      let riskClass = 'low';
      if ((agg.risk_score || 0) >= 0.66) riskClass = 'high';
      else if ((agg.risk_score || 0) >= 0.33) riskClass = 'medium';

      return {
        cohortSize: count,
        predictedClass,
        confidence: Number(confidence.toFixed(4)),
        readmissionProbability: Number(readmissionProbability.toFixed(4)),
        riskScore: Number((agg.risk_score || 0).toFixed(4)),
        riskClass,
        probabilities: {
          no: Number(probNo.toFixed(4)),
          lt30: Number(probLt30.toFixed(4)),
          gt30: Number(probGt30.toFixed(4)),
        },
        note,
      };
    };

    let promptPrediction = buildPrediction(total, where, params);

    // If exact filter has no rows, relax strict outcome constraints and still
    // return a meaningful model prediction for the prompt context.
    if (!promptPrediction) {
      const relaxedFilters = { ...(query.filters || {}) };
      delete relaxedFilters.risk_class;
      delete relaxedFilters.readmitted;

      const relaxedConditions = [];
      const relaxedParams = [];
      for (const [key, val] of Object.entries(relaxedFilters)) {
        if (val === undefined || val === null || val === '') continue;
        if (Array.isArray(val)) {
          relaxedConditions.push(`${key} IN (${val.map(() => '?').join(',')})`);
          relaxedParams.push(...val);
        } else if (typeof val === 'string' && val.includes('%')) {
          relaxedConditions.push(`${key} LIKE ?`);
          relaxedParams.push(val);
        } else if (typeof val === 'string' && /^[><=]+\d+/.test(val)) {
          const op = val.match(/^[><=]+/)[0];
          const num = Number(val.slice(op.length));
          relaxedConditions.push(`${key} ${op} ?`);
          relaxedParams.push(num);
        } else {
          relaxedConditions.push(`${key} = ?`);
          relaxedParams.push(val);
        }
      }

      const relaxedWhere = relaxedConditions.length
        ? `WHERE ${relaxedConditions.join(' AND ')}`
        : '';
      const relaxedCountStmt = db.prepare(`SELECT COUNT(*) as total FROM patients ${relaxedWhere}`);
      const relaxedTotal = relaxedCountStmt.get(...relaxedParams).total;
      promptPrediction = buildPrediction(
        relaxedTotal,
        relaxedWhere,
        relaxedParams,
        'No exact matches for all constraints; showing prediction on closest matching cohort.'
      );
    }

    const rowsStmt = db.prepare(
      `SELECT * FROM patients ${where} ORDER BY ${sortBy} ${order} LIMIT ? OFFSET ?`
    );
    const rows = rowsStmt.all(...params, Number(limit), Number(offset));
    db.close();

    res.json({
      prompt,
      query,
      promptPrediction,
      data: rows,
      pagination: { page: Number(page), limit: Number(limit), total, pages: Math.ceil(total / limit) },
    });
  } catch (err) {
    console.error('Execute query failed:', err);
    res.status(500).json({ error: 'Failed to execute query' });
  }
});

module.exports = router;
