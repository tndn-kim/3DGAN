// This file is loaded by index.html.
// Replace this object with actual Pipeline.py output using tools/export_dashboard_results.py.
window.STRUCTURED_RESULTS = {
  actual: false,
  generatedAt: "sample-schema",
  models: ["rf", "xgb", "bigru"],
  modelLabels: {
    rf: "Random Forest",
    xgb: "XGBoost",
    bigru: "BiGRU"
  },
  classLabels: [
    "Benign",
    "Analysis & Reconnaissance",
    "System Compromise",
    "DoS & Fuzzers"
  ],
  classCountsBefore: [358332, 17120, 38383, 34080],
  classCountsAfter: [358332, 358332, 358332, 358332],
  before: {
    rf: { metrics: { accuracy: 0.932, f1_macro: 0.906, micro_FNR: 0.103 } },
    xgb: { metrics: { accuracy: 0.948, f1_macro: 0.927, micro_FNR: 0.076 } },
    bigru: { metrics: { accuracy: 0.956, f1_macro: 0.940, micro_FNR: 0.062 } }
  },
  after: {
    rf: { metrics: { accuracy: 0.958, f1_macro: 0.944, micro_FNR: 0.055 } },
    xgb: { metrics: { accuracy: 0.972, f1_macro: 0.961, micro_FNR: 0.037 } },
    bigru: { metrics: { accuracy: 0.978, f1_macro: 0.973, micro_FNR: 0.019 } }
  }
};
