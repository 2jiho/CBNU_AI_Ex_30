"""
4. 모폴로지 필터
영상을 이진화한 후에 사용자로부터 Erosion, Dilation, Opening, Closing에 대한 
선택과 횟수를 입력받아서 해당 결과를 출력하시오.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./TestImage/Lena.png", cv2.IMREAD_GRAYSCALE)
otsu_thr, otus_mask = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binary = otus_mask * 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

seleted_process = input("selet Erosion, Dilation, Opening, Closing[e/d/o/c]:")
seleted_process = seleted_process.upper()
seleted_loop = int(input("loop:"))
for i in range(seleted_loop):
    if seleted_process in ["E", "O"]:
        binary = cv2.erode(binary, kernel)
    elif seleted_process in ["D", "C"]:
        binary = cv2.dilate(binary, kernel)

if seleted_process in ["O", "C"]:
    for i in range(seleted_loop):
        if seleted_process == "C":
            binary = cv2.erode(binary, kernel)
        elif seleted_process == "O":
            binary = cv2.dilate(binary, kernel)
cv2.imshow("binary", binary)
cv2.waitKey()
