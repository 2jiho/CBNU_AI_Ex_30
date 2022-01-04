import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread("lena.png", 0)
cv2.imshow("gray", gray)
cv2.waitKey(0)

hist, bins = np.histogram(gray, 256, [0, 255])
plt.fill(hist)
plt.xlabel("pixel value")
plt.show()

gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel("pixel value")
plt.show()

cv2.imshow("gray_eq", gray_eq)
cv2.waitKey(0)

color = cv2.imread("lena.png")
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("color", color)
cv2.imshow("color_eq", color_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
