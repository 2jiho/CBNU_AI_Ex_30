import cv2

print(f"opencv version: {cv2.__version__}")

img_path = "lena.png"

img = cv2.imread(img_path)
assert img is not None
print(f"shape: {img.shape}")
print(f"dtype: {img.dtype}")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
assert img is not None
print(f"shape: {img.shape}")
print(f"dtype: {img.dtype}")
