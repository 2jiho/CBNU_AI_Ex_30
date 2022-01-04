"""
2. 공간 도메인 필터링
각 픽셀에 임의의 값을 더해 노이즈를 생성하고, 사용자로부터 Bilateral filtering을 
위한 diameter, SigmaColor, SigmaSpace를 입력받아 노이즈를 제거하고 노이즈 제거 
전후의 영상을 출력하시오. 
(다양한 파라미터 변화를 통해 영상이 어떻게 변화하는지 보고서에 넣으시오.)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./TestImage/Lena.png").astype(np.float32) / 255
noised = image + 0.2 * np.random.rand(*image.shape).astype(np.float32)
noised = noised.clip(0, 1)
plt.imshow(noised[:, :, [2, 1, 0]])
plt.show()

"""
diameter: 필터링에 사용될 이웃 픽셀의 지름, -1을 입력하면 sigmaSpace 값에 의해 자동 결정
sigmaColor: 색 공간에서 필터의 표준 편차
sigmaSpace: 좌표 공간에서 필터의 표준 편차
"""
diameter = input("diameter:")
sigmaColor = input("sigmaColor:")
sigmaSpace = input("sigmaSpace:")
bilat = cv2.bilateralFilter(noised, int(diameter), float(sigmaColor), float(sigmaSpace))
plt.imshow(bilat[:, :, [2, 1, 0]])
plt.show()
