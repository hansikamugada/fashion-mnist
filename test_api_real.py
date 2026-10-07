import requests
import json
import base64

real_samples = json.load(open('static/data/real_catalog.json'))
print("=== TESTING FLASK API /api/predict ON REAL-WORLD SAMPLES (CURRENT SERVER) ===")
real_pass = True
for r in real_samples:
    with open('static/images/real_samples/' + r['filename'], 'rb') as f:
        b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
    resp = requests.post('http://127.0.0.1:5000/api/predict', json={
        'image': b64,
        'invert_mode': 'auto'
    })
    res = resp.json()
    pred = res.get('prediction')
    conf = res.get('confidence')
    match = (pred == r['label'])
    if not match: real_pass = False
    print(f"{r['label']:12} -> {pred:12} ({conf:5.1f}%) | {'PASS' if match else 'FAIL'}")

print("\nOverall Real-World API Accuracy:", "100% PERFECT!" if real_pass else "ERRORS FOUND!")
