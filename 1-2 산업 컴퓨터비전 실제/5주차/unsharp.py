import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

image = cv2.imread("../lena.png")

KSIZE = 11
ALPHA = 20
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE // 2, KSIZE // 2] += 1 + ALPHA
print(kernel.shape, kernel.dtype, kernel.sum())

filtered = cv2.filter2D(image, -1, kernel)


plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.axis("off")
plt.title("image")
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis("off")
plt.title("filtered")
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout(True)

cv2.imshow("before", image)
cv2.imshow("after", filtered)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
