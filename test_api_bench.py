import requests
import json
import base64

samples = json.load(open('static/data/sample_catalog.json'))
print("=== TESTING FLASK API /api/predict ON BENCHMARK SAMPLES ===")
all_pass = True
for s in samples:
    with open('static/images/samples/' + s['filename'], 'rb') as f:
        b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')
    resp = requests.post('http://127.0.0.1:5000/api/predict', json={
        'image': b64,
        'invert_mode': 'auto'
    })
    res = resp.json()
    pred = res.get('prediction')
    conf = res.get('confidence')
    match = (pred == s['label'])
    if not match: all_pass = False
    print(f"{s['label']:12} -> {pred:12} ({conf:5.1f}%) | {'PASS' if match else 'FAIL'}")

print("\nOverall Benchmark API Accuracy:", "100% PERFECT!" if all_pass else "ERRORS FOUND!")
