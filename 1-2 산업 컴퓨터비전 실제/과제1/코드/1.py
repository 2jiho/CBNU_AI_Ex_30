"""
1. 히스토그램 평탄화
사용자로부터 R, G, B 중의 하나의 채널을 입력받고 입력받은 채널에 대한 
히스토그램을 그리고 평탄화를 한 후에 그 영상을 출력하시오. 
(선택받은 채널 이외의 채널 값은 변화하지 않음)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 채널 선택
input_value = input("Select (R/G/B):")
input_value = input_value.upper()
selected_channel = 2 if input_value == "R" else 1 if input_value == "G" else 0
# 이미지 불러와서 출력
color = cv2.imread("./TestImage/Lena.png")
cv2.imshow("color org", color)
cv2.waitKey(1)
# 선택된 채널 히스토그램 그리기
hist, bins = np.histogram(color[..., selected_channel], 256, [0, 255])
plt.subplot(211)
plt.title("histogram")
plt.fill_between(range(256), hist, 0)
# 평탄화와 평탄화된 히스토그램 그리기
color_eq = color.copy()
color_eq[..., selected_channel] = cv2.equalizeHist(color[..., selected_channel])
hist, bins = np.histogram(color_eq[..., selected_channel], 256, [0, 255])
plt.subplot(212)
plt.title("equalizeHist")
plt.fill_between(range(256), hist, 0)
plt.tight_layout()
plt.show(block=False)
# 평탄화 이미지 출력
cv2.imshow("color eq", color_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
