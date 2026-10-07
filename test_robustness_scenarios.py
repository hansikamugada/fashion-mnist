import requests
import json
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

print("=== TESTING ROBUSTNESS SCENARIOS ===")

def test_modified_image(img_path, label, transform_fn):
    img = Image.open(img_path)
    transformed = transform_fn(img)
    buf = io.BytesIO()
    transformed.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
    resp = requests.post('http://127.0.0.1:5000/api/predict', json={'image': b64, 'invert_mode': 'auto'})
    res = resp.json()
    pred = res.get('prediction')
    conf = res.get('confidence')
    match = (pred == label)
    print(f"{label:12} [{transform_fn.__name__:18}] -> {pred:12} ({conf:5.1f}%) | {'PASS' if match else 'FAIL'}")
    return match

# Test 1: Camera photo with warm color shift
def warm_light(im):
    arr = np.array(im, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.1, 0, 255) # warmer red
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.9, 0, 255) # lower blue
    return Image.fromarray(arr.astype(np.uint8))

# Test 2: High contrast photo
def high_contrast(im):
    enh = ImageEnhance.Contrast(im)
    return enh.enhance(1.4)

# Test 3: Camera photo with slight blur
def camera_blur(im):
    return im.filter(ImageFilter.GaussianBlur(radius=1.5))

# Test 4: Downscaled low-res webcam image (320x240)
def webcam_lowres(im):
    return im.resize((320, 320), Image.Resampling.BILINEAR)

passed = 0
total = 0
for fname, lbl in [
    ('real_0_tshirt.jpg', 'T-shirt/top'),
    ('real_1_trouser.jpg', 'Trouser'),
    ('real_7_sneaker.jpg', 'Sneaker'),
    ('real_8_bag.jpg', 'Bag'),
    ('real_9_ankle_boot.jpg', 'Ankle boot')
]:
    for fn in [warm_light, high_contrast, camera_blur, webcam_lowres]:
        total += 1
        if test_modified_image('static/images/real_samples/' + fname, lbl, fn):
            passed += 1

print(f"\nRobustness Score: {passed} / {total} ({passed/total*100:.1f}%)")
