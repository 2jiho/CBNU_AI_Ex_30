import cv2

images = []
images.append(cv2.imread("0.jpeg", cv2.IMREAD_COLOR))
images.append(cv2.imread("1.jpeg", cv2.IMREAD_COLOR))

stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(images)

if ret == cv2.STITCHER_OK:
    pano = cv2.resize(pano, dsize=(0, 0), fx=0.2, fy=0.2)
    cv2.imshow("pano", pano)
    cv2.waitKey()

    cv2.destroyAllWindows()

else:
    print("Error")
