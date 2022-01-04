import cv2
import numpy as np

roi_height = 0


def mouse_event(event, x, y, flags, param):
    global roi_height

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_height = y


cv2.namedWindow("show")
cv2.setMouseCallback("show", mouse_event)

while True:
    # Load
    img = cv2.imread("40.jpg")

    # Blur
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Gray
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Edge
    canny = cv2.Canny(gray, 50, 200)

    # Sky ROI
    canny[0:roi_height, :] = 0
    cv2.line(blur, (0, roi_height), (blur.shape[1], roi_height), (255, 255, 0), 3)

    # Hough TF
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength, maxLineGap)
    if lines is not None:
        for index, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            try:
                slope = (y1 - y2) / (x1 - x2)
            except ZeroDivisionError:
                slope = 0

            # skip horizontal line
            if -0.5 < slope < 0.5:
                continue

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Show
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    concat_img = cv2.hconcat([img, blur, canny])
    cv2.imshow("show", concat_img)
    if cv2.waitKey(33) == 27:
        break
