import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../lena.png").astype(np.float32) / 255.0
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

data = image_lab.reshape((-1, 3))

num_classes = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
_, labels, centers = cv2.kmeans(
    data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2RGB)

plt.subplot(121)
plt.axis("off")
plt.title("orignal")
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis("off")
plt.title("segmented")
plt.imshow(segmented)
plt.tight_layout()
plt.show()
