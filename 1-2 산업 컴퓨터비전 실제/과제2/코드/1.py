"""
1. Feature Detection
- stitching.zip에서 4장의 영상(boat1, budapest1, newpaper1, s1)을 선택한 후에 Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드를 작성하시오.
"""
import cv2
import numpy as np

imgs = []
imgs.append(cv2.imread("./stitching/boat1.jpg"))
imgs.append(cv2.imread("./stitching/budapest1.jpg"))
imgs.append(cv2.imread("./stitching/newspaper1.jpg"))
imgs.append(cv2.imread("./stitching/s1.jpg"))

result_imgs = []
for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 200)
    corners = cv2.cornerHarris(canny, 2, 3, 0.04)
    show_img = np.copy(img)
    show_img[corners > 0.1 * corners.max()] = [255, 0, 255]
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    show_img = cv2.hconcat([canny, show_img])
    result_imgs.append(show_img)


for index, result_img in enumerate(result_imgs):
    cv2.imshow("img", result_img)
    cv2.imwrite(f"./1_{index+1}.png", result_img)
    cv2.waitKey()

cv2.destroyAllWindows()
