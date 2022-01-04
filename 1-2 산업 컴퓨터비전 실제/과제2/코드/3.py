"""
3. Panorama
- CreaterStitcher 함수를 이용하여 4개의 영상 셋에 대해서 파노라마 이미지를 만드는 방법을 구현하시오.
"""
import cv2
import numpy as np

imgs_list = []

imgs_list.append([])
for i in range(1, 7):
    imgs_list[-1].append(cv2.imread(f"./stitching/boat{i}.jpg"))

imgs_list.append([])
for i in range(1, 7):
    imgs_list[-1].append(cv2.imread(f"./stitching/budapest{i}.jpg"))

imgs_list.append([])
for i in range(1, 5):
    imgs_list[-1].append(cv2.imread(f"./stitching/newspaper{i}.jpg"))

imgs_list.append([])
for i in range(1, 3):
    imgs_list[-1].append(cv2.imread(f"./stitching/s{i}.jpg"))

result_imgs = []

stitcher = cv2.createStitcher()
for index, imgs in enumerate(imgs_list):
    ret, pano = stitcher.stitch(imgs)
    if ret == cv2.STITCHER_OK:
        # pano = cv2.resize(pano, dsize=(0, 0), fx=0.2, fy=0.2)
        result_imgs.append(pano)
    else:
        print(index + 1, "img set Error", ret)

for index, result_img in enumerate(result_imgs):
    cv2.imwrite(f"./3_{index+1}.png", result_img)
    cv2.imshow("img", result_img)
    cv2.waitKey()

cv2.destroyAllWindows()
