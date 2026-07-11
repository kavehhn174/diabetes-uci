const Database = require('better-sqlite3');
const path = require('path');

const DB_PATH = process.env.DB_PATH || path.join(__dirname, '..', 'data.db');

function getDb() {
  const db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  return db;
}

function initSchema(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS patients (
      encounter_id INTEGER PRIMARY KEY,
      patient_nbr INTEGER,
      race TEXT,
      gender TEXT,
      age TEXT,
      weight TEXT,
      admission_type_id INTEGER,
      discharge_disposition_id INTEGER,
      admission_source_id INTEGER,
      time_in_hospital INTEGER,
      payer_code TEXT,
      medical_specialty TEXT,
      num_lab_procedures INTEGER,
      num_procedures INTEGER,
      num_medications INTEGER,
      number_outpatient INTEGER,
      number_emergency INTEGER,
      number_inpatient INTEGER,
      diag_1 TEXT,
      diag_2 TEXT,
      diag_3 TEXT,
      number_diagnoses INTEGER,
      glucose_test_result TEXT,
      a1c_test_result TEXT,
      metformin TEXT,
      repaglinide TEXT,
      nateglinide TEXT,
      chlorpropamide TEXT,
      glimepiride TEXT,
      acetohexamide TEXT,
      glipizide TEXT,
      glyburide TEXT,
      tolbutamide TEXT,
      pioglitazone TEXT,
      rosiglitazone TEXT,
      acarbose TEXT,
      miglitol TEXT,
      troglitazone TEXT,
      tolazamide TEXT,
      examide TEXT,
      citoglipton TEXT,
      insulin TEXT,
      glyburide_metformin TEXT,
      glipizide_metformin TEXT,
      glimepiride_pioglitazone TEXT,
      metformin_rosiglitazone TEXT,
      metformin_pioglitazone TEXT,
      change TEXT,
      diabetesMed TEXT,
      readmitted TEXT,
      prob_no REAL,
      prob_lt30 REAL,
      prob_gt30 REAL,
      predicted_class TEXT,
      risk_score REAL,
      risk_class TEXT,
      risk_label TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_patients_readmitted ON patients(readmitted);
    CREATE INDEX IF NOT EXISTS idx_patients_risk_class ON patients(risk_class);
    CREATE INDEX IF NOT EXISTS idx_patients_age ON patients(age);
    CREATE INDEX IF NOT EXISTS idx_patients_gender ON patients(gender);
    CREATE INDEX IF NOT EXISTS idx_patients_race ON patients(race);

    CREATE TABLE IF NOT EXISTS dataset_stats (
      key TEXT PRIMARY KEY,
      value TEXT
    );
  `);
}

module.exports = { getDb, initSchema, DB_PATH };
