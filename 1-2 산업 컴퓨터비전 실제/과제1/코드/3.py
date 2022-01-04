"""
3. 주파수 도메인 필터링
DFT를 통해서 영상을 주파수 도메인으로 바꿔서 출력 한 후에 사용자로부터 
반지름을 입력받아서 그 크기만큼의 원을 그린 후에 DFT 결과에 곱해준 후에 
IDFT를 해서 필터링된 영상을 출력하시오. 사용자로부터 Low pass인지 High 
Pass인지를 입력받아 Low pass면 원 안을 통과시키고, High Pass면 원 바깥을 
통과시키도록 하시오.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./TestImage/Lena.png", 0).astype(np.float32) / 255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
mask = np.zeros(fft.shape, np.uint8)
radius = input("radius:")
cv2.circle(mask, (image.shape[1] // 2, image.shape[0] // 2), int(radius), (1, 1, 1), -1)
low_high = input("select Low pass, High pass[low/high]:")
low_high = low_high.upper()
if low_high == "HIGH":
    mask = 1 - mask
fft_shift *= mask
fft = np.fft.fftshift(fft_shift, axes=[0, 1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

plt.figure()
plt.subplot(131)
plt.axis("off")
plt.title("org")
plt.imshow(image, cmap="gray")
plt.subplot(132)
plt.axis("off")
plt.title("filtered")
plt.imshow(filtered, cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("mask")
plt.imshow(mask_new * 255, cmap="gray")
plt.tight_layout(True)
plt.show()
