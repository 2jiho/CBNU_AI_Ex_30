import cv2

print(f"opencv version: {cv2.__version__}")

img_path = "lena.png"
img = cv2.imread(img_path)
cv2.imshow("img1", img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img2", img_gray)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("img3", img_hsv)
img_hsv[:, :, 2] *= 2
img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("img4", img_bgr)
key = cv2.waitKey(0)
