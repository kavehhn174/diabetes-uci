const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const csv = require('csv-parser');
const { getDb, initSchema } = require('../src/database');

const CSV_PATH = process.env.CSV_PATH || path.join(__dirname, '..', '..', 'diabetic_data.csv');
const PREDICTIONS_PATH = path.join(__dirname, '..', 'model', 'predictions.csv');
const MODEL_DIR = path.join(__dirname, '..', 'model');
const GENERATE_PREDICTIONS_SCRIPT = path.join(MODEL_DIR, 'generate_predictions.py');
const BATCH_SIZE = 2000;

function snakeCase(name) {
  return name.replace(/\-/g, '_').toLowerCase();
}

function cleanRow(row) {
  const out = {};
  for (const [key, value] of Object.entries(row)) {
    const k = snakeCase(key);
    let v = value;
    if (v === '?' || v === '' || v === undefined || v === null) {
      v = null;
    } else if (
      [
        'encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses',
      ].includes(k)
    ) {
      v = Number(v);
    }
    out[k] = v;
  }
  return out;
}

async function loadRows(filePath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => rows.push(cleanRow(row)))
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

async function loadPredictions() {
  return new Promise((resolve, reject) => {
    const rows = {};
    if (!fs.existsSync(PREDICTIONS_PATH)) {
      return resolve(rows);
    }
    fs.createReadStream(PREDICTIONS_PATH)
      .pipe(csv())
      .on('data', (row) => {
        const parseNum = (value) => {
          if (value === undefined || value === null || value === '') return null;
          const n = Number(value);
          return Number.isFinite(n) ? n : null;
        };
        const id = Number(row.encounter_id);
        const probNo = parseNum(row['prob_NO']);
        const probLt30 = parseNum(row['prob_<30']);
        const probGt30 = parseNum(row['prob_>30']);
        let riskScore = parseNum(row.risk_score);
        if (riskScore === null && probLt30 !== null && probGt30 !== null) {
          riskScore = Number((0.7 * probLt30 + 0.2 * probGt30).toFixed(4));
        }
        rows[id] = {
          prob_no: probNo,
          prob_lt30: probLt30,
          prob_gt30: probGt30,
          predicted_class: row.predicted_class || null,
          risk_score: riskScore,
          risk_class: row.risk_class || null,
          risk_label: row.risk_label || null,
        };
      })
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

function getPythonCandidates() {
  const candidates = [];
  if (process.env.PYTHON_BIN) {
    candidates.push(process.env.PYTHON_BIN);
  }
  const venvPython = path.join(MODEL_DIR, 'venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) {
    candidates.push(venvPython);
  }
  candidates.push('python3', 'python');
  return candidates;
}

function regeneratePredictionsFromModel() {
  const candidates = getPythonCandidates();
  const failures = [];

  for (const python of candidates) {
    const result = spawnSync(python, [GENERATE_PREDICTIONS_SCRIPT], {
      cwd: MODEL_DIR,
      stdio: 'inherit',
    });

    if (result.status === 0) {
      return;
    }
    failures.push(`${python} (exit ${result.status ?? 'null'})`);
  }

  throw new Error(
    `Failed to generate Bayesian predictions using ${GENERATE_PREDICTIONS_SCRIPT}. Tried: ${failures.join(', ')}`
  );
}

async function seed() {
  console.log(`Loading CSV from ${CSV_PATH}...`);
  const rows = await loadRows(CSV_PATH);
  console.log(`Loaded ${rows.length.toLocaleString()} rows`);

  console.log('Generating predictions from trained Bayesian model (bn_model.pkl)...');
  regeneratePredictionsFromModel();

  const predictions = await loadPredictions();
  const predictionCount = Object.keys(predictions).length;
  if (!predictionCount) {
    throw new Error(
      `No Bayesian predictions found in ${PREDICTIONS_PATH}. Seeding must use model-generated predictions only.`
    );
  }
  console.log(`Loaded ${predictionCount.toLocaleString()} model predictions from ${PREDICTIONS_PATH}`);

  const missingPredictionIds = [];
  const enrichedRows = rows.map((row) => {
    const pred = predictions[row.encounter_id];
    if (!pred || pred.risk_score === null) {
      missingPredictionIds.push(row.encounter_id);
    }
    return { ...row, ...pred };
  });

  if (missingPredictionIds.length > 0) {
    const sample = missingPredictionIds.slice(0, 10).join(', ');
    throw new Error(
      `Missing Bayesian predictions for ${missingPredictionIds.length} encounters (sample IDs: ${sample}).`
    );
  }

  const db = getDb();
  initSchema(db);

  console.log('Resetting patients table...');
  db.exec('DELETE FROM patients');

  const columns = [
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
  const placeholders = columns.map(() => '?').join(',');
  const insert = db.prepare(`INSERT INTO patients (${columns.join(',')}) VALUES (${placeholders})`);

  const insertMany = db.transaction((batch) => {
    for (const row of batch) {
      insert.run(columns.map((c) => row[c] ?? null));
    }
  });

  for (let i = 0; i < enrichedRows.length; i += BATCH_SIZE) {
    const batch = enrichedRows.slice(i, i + BATCH_SIZE);
    insertMany(batch);
    process.stdout.write(`\rInserted ${Math.min(i + BATCH_SIZE, enrichedRows.length).toLocaleString()} / ${enrichedRows.length.toLocaleString()} rows`);
  }
  console.log('');

  db.prepare('INSERT OR REPLACE INTO dataset_stats (key, value) VALUES (?, ?)').run('total_patients', String(rows.length));
  db.prepare('INSERT OR REPLACE INTO dataset_stats (key, value) VALUES (?, ?)').run('model_predictions', 'true');

  db.close();
  console.log('Seeding complete.');
}

seed().catch((err) => {
  console.error(err);
  process.exit(1);
});
