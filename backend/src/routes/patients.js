const express = require('express');
const { getDb } = require('../database');

const router = express.Router();

const ALLOWED_COLUMNS = [
  'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id',
  'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code',
  'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications',
  'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
  'number_diagnoses', 'glucose_test_result', 'a1c_test_result', 'metformin', 'repaglinide',
  'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
  'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
  'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide_metformin', 'glipizide_metformin',
  'glimepiride_pioglitazone', 'metformin_rosiglitazone', 'metformin_pioglitazone', 'change',
  'diabetesMed', 'readmitted', 'prob_no', 'prob_lt30', 'prob_gt30', 'predicted_class',
  'risk_score', 'risk_class', 'risk_label',
];

const TEXT_COLUMNS = [
  'race', 'gender', 'age', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
  'glucose_test_result', 'a1c_test_result', 'metformin', 'repaglinide', 'nateglinide',
  'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
  'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
  'examide', 'citoglipton', 'insulin', 'glyburide_metformin', 'glipizide_metformin',
  'glimepiride_pioglitazone', 'metformin_rosiglitazone', 'metformin_pioglitazone', 'change',
  'diabetesMed', 'readmitted', 'predicted_class', 'risk_class', 'risk_label',
];

function buildWhere(filters) {
  const conditions = [];
  const params = [];

  for (const [key, val] of Object.entries(filters)) {
    if (!ALLOWED_COLUMNS.includes(key)) continue;
    if (val === undefined || val === null || val === '') continue;

    if (Array.isArray(val)) {
      if (TEXT_COLUMNS.includes(key)) {
        conditions.push(`${key} IN (${val.map(() => '?').join(',')})`);
        params.push(...val);
      } else {
        conditions.push(`${key} IN (${val.map(() => '?').join(',')})`);
        params.push(...val.map(Number));
      }
    } else if (typeof val === 'object' && ('min' in val || 'max' in val)) {
      if (val.min !== undefined && val.min !== '') {
        conditions.push(`${key} >= ?`);
        params.push(Number(val.min));
      }
      if (val.max !== undefined && val.max !== '') {
        conditions.push(`${key} <= ?`);
        params.push(Number(val.max));
      }
    } else if (typeof val === 'string' && (val.startsWith('>') || val.startsWith('<') || val.startsWith('='))) {
      const op = val.startsWith('>=') || val.startsWith('<=') ? val.slice(0, 2) : val[0];
      const num = Number(val.slice(op.length));
      conditions.push(`${key} ${op} ?`);
      params.push(num);
    } else if (TEXT_COLUMNS.includes(key)) {
      conditions.push(`${key} = ?`);
      params.push(val);
    } else {
      conditions.push(`${key} = ?`);
      params.push(Number(val));
    }
  }

  return { conditions, params };
}

// GET /api/patients/search
router.get('/search', (req, res) => {
  const db = getDb();
  const page = Math.max(1, Number(req.query.page || 1));
  const limit = Math.min(100, Math.max(1, Number(req.query.limit || 20)));
  const offset = (page - 1) * limit;
  const sortBy = ALLOWED_COLUMNS.includes(req.query.sortBy) ? req.query.sortBy : 'risk_score';
  const order = req.query.order === 'asc' ? 'ASC' : 'DESC';

  const filters = req.query.filters ? JSON.parse(req.query.filters) : {};
  const { conditions, params } = buildWhere(filters);
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';

  const countStmt = db.prepare(`SELECT COUNT(*) as total FROM patients ${where}`);
  const { total } = countStmt.get(...params);

  const rowsStmt = db.prepare(
    `SELECT * FROM patients ${where} ORDER BY ${sortBy} ${order} LIMIT ? OFFSET ?`
  );
  const rows = rowsStmt.all(...params, limit, offset);

  db.close();
  res.json({
    data: rows,
    pagination: { page, limit, total, pages: Math.ceil(total / limit) },
  });
});

// GET /api/patients/:id
router.get('/:id', (req, res) => {
  const db = getDb();
  const row = db.prepare('SELECT * FROM patients WHERE encounter_id = ?').get(req.params.id);
  db.close();
  if (!row) return res.status(404).json({ error: 'Patient not found' });
  res.json(row);
});

// POST /api/patients/predict
router.post('/predict', (req, res) => {
  const db = getDb();
  const { encounter_id } = req.body;
  if (!encounter_id) {
    db.close();
    return res.status(400).json({ error: 'encounter_id is required' });
  }
  const row = db.prepare('SELECT * FROM patients WHERE encounter_id = ?').get(encounter_id);
  db.close();
  if (!row) return res.status(404).json({ error: 'Patient not found' });

  res.json({
    encounter_id: row.encounter_id,
    predicted_class: row.predicted_class,
    actual_class: row.readmitted,
    risk_score: row.risk_score,
    risk_class: row.risk_class,
    risk_label: row.risk_label,
    probabilities: {
      no: row.prob_no,
      lt30: row.prob_lt30,
      gt30: row.prob_gt30,
    },
    top_features: {
      number_inpatient: row.number_inpatient,
      discharge_disposition_id: row.discharge_disposition_id,
      number_emergency: row.number_emergency,
      number_diagnoses: row.number_diagnoses,
      medical_specialty: row.medical_specialty,
      admission_source_id: row.admission_source_id,
      number_outpatient: row.number_outpatient,
      num_lab_procedures: row.num_lab_procedures,
    },
  });
});

module.exports = router;
