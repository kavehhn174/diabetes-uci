const express = require('express');
const { getDb } = require('../database');

const router = express.Router();

router.get('/summary', (req, res) => {
  const db = getDb();

  const total = db.prepare('SELECT COUNT(*) as count FROM patients').get().count;
  const readmitted = db.prepare(
    "SELECT COUNT(*) as count FROM patients WHERE readmitted != 'NO'"
  ).get().count;
  const within30 = db.prepare(
    "SELECT COUNT(*) as count FROM patients WHERE readmitted = '<30'"
  ).get().count;
  const riskDistribution = db.prepare(
    'SELECT risk_class, COUNT(*) as count FROM patients GROUP BY risk_class'
  ).all();
  const ageDistribution = db.prepare(
    "SELECT age, COUNT(*) as count FROM patients GROUP BY age ORDER BY CASE age " +
      "WHEN '[0-10)' THEN 1 WHEN '[10-20)' THEN 2 WHEN '[20-30)' THEN 3 WHEN '[30-40)' THEN 4 " +
      "WHEN '[40-50)' THEN 5 WHEN '[50-60)' THEN 6 WHEN '[60-70)' THEN 7 WHEN '[70-80)' THEN 8 " +
      "WHEN '[80-90)' THEN 9 WHEN '[90-100)' THEN 10 ELSE 11 END"
  ).all();
  const readmittedByAge = db.prepare(
    "SELECT age, readmitted, COUNT(*) as count FROM patients GROUP BY age, readmitted " +
      "ORDER BY CASE age WHEN '[0-10)' THEN 1 WHEN '[10-20)' THEN 2 WHEN '[20-30)' THEN 3 " +
      "WHEN '[30-40)' THEN 4 WHEN '[40-50)' THEN 5 WHEN '[50-60)' THEN 6 WHEN '[60-70)' THEN 7 " +
      "WHEN '[70-80)' THEN 8 WHEN '[80-90)' THEN 9 WHEN '[90-100)' THEN 10 ELSE 11 END"
  ).all();
  const genderDistribution = db.prepare(
    'SELECT gender, COUNT(*) as count FROM patients GROUP BY gender'
  ).all();
  const raceDistribution = db.prepare(
    'SELECT race, COUNT(*) as count FROM patients GROUP BY race'
  ).all();
  const topSpecialties = db.prepare(
    "SELECT medical_specialty, COUNT(*) as count FROM patients WHERE medical_specialty IS NOT NULL AND medical_specialty != '?' GROUP BY medical_specialty ORDER BY count DESC LIMIT 15"
  ).all();
  const medicationStats = db.prepare(
    "SELECT 'diabetesMed' as key, diabetesMed as value, COUNT(*) as count FROM patients GROUP BY diabetesMed UNION ALL " +
      "SELECT 'insulin', insulin, COUNT(*) FROM patients GROUP BY insulin UNION ALL " +
      "SELECT 'change', change, COUNT(*) FROM patients GROUP BY change"
  ).all();
  const avgHospitalTime = db.prepare(
    'SELECT ROUND(AVG(time_in_hospital), 2) as avg_time FROM patients'
  ).get();
  const avgRiskScore = db.prepare(
    'SELECT ROUND(AVG(risk_score), 4) as avg_risk FROM patients'
  ).get();
  const topFeatures = db.prepare(
    'SELECT ROUND(AVG(number_inpatient), 2) as avg_inpatient, ' +
      'ROUND(AVG(number_emergency), 2) as avg_emergency, ' +
      'ROUND(AVG(number_outpatient), 2) as avg_outpatient, ' +
      'ROUND(AVG(num_lab_procedures), 2) as avg_lab_procedures, ' +
      'ROUND(AVG(num_medications), 2) as avg_medications, ' +
      'ROUND(AVG(number_diagnoses), 2) as avg_diagnoses FROM patients'
  ).get();
  const modelPredictions = db.prepare(
    "SELECT value FROM dataset_stats WHERE key = 'model_predictions'"
  ).get();

  db.close();
  res.json({
    modelPredictions: modelPredictions?.value === 'true',
    total,
    readmitted,
    within30,
    readmissionRate: Number((readmitted / total).toFixed(4)),
    within30Rate: Number((within30 / total).toFixed(4)),
    riskDistribution,
    ageDistribution,
    readmittedByAge,
    genderDistribution,
    raceDistribution,
    topSpecialties,
    medicationStats,
    avgHospitalTime: avgHospitalTime.avg_time,
    avgRiskScore: avgRiskScore.avg_risk,
    topFeatures,
  });
});

router.get('/readmitted-by-feature/:feature', (req, res) => {
  const feature = req.params.feature;
  const allowed = [
    'age', 'gender', 'race', 'medical_specialty', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'diabetesMed', 'insulin', 'change',
  ];
  if (!allowed.includes(feature)) {
    return res.status(400).json({ error: 'Invalid feature' });
  }

  const db = getDb();
  const rows = db.prepare(
    `SELECT ${feature} as label, readmitted, COUNT(*) as count FROM patients WHERE ${feature} IS NOT NULL AND ${feature} != '?' GROUP BY ${feature}, readmitted ORDER BY count DESC`
  ).all();
  db.close();
  res.json(rows);
});

module.exports = router;
