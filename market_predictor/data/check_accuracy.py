import sys, joblib
sys.path.insert(0, 'market_predictor')

info = joblib.load('market_predictor/models/saved/model_info.pkl')
print("Model type:", info.get('model_type'))
print("Train samples:", info.get('train_samples'))
print("Test samples:", info.get('test_samples'))
print("Test cutoff:", info.get('test_cutoff'))
print()

cr = info.get('classification_report', {})
print("=== Per-Class Performance ===")
for label in ['HIGH', 'MEDIUM', 'LOW']:
    m = cr.get(label, {})
    p = m.get('precision', 0)
    r = m.get('recall', 0)
    f = m.get('f1-score', 0)
    print(f"  {label:8s}  precision={p:.0%}  recall={r:.0%}  f1={f:.0%}")

print()
acc = cr.get('accuracy', 0)
print(f"Overall accuracy: {acc:.0%}")
print()
print("=== What this means ===")
print("Precision = when model says HIGH, how often it's right")
print("Recall    = of all actual HIGH events, how many did model catch")
print()
print("For trading use, HIGH precision matters most.")
print("If HIGH precision > 60%, the model is useful for flagging impactful news.")
