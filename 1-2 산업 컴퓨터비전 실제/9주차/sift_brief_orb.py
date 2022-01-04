import cv2

img = cv2.imread("img.jpeg")

# surf = cv2.xfeatures2d.SURF_create(10000)
# surf.setExtended(True)
# surf.setNOcatves(3)
# surf.setNOcatvesLayers(10)
# surf.setUpright(False)

# keyPoints, descriptors = surf.detectAndCompute(img, None)

# show_img = cv2.drawKeypoints(img, keypoints, None)

# cv2.imshow("surf", show_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create(50)
keyPoints, descriptors = sift.detectAndCompute(img, None)

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32, True)

keyPoints, descriptors = brief.compute(img, keyPoints)

show_img = cv2.drawKeypoints(
    img, keyPoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("brief", show_img)
cv2.waitKey()
cv2.destroyAllWindows()

orb = cv2.ORB_create()
orb.setMaxFeatures(200)

keyPoints = orb.detect(img, None)
keyPoints, descriptors = orb.compute(img, keyPoints)
show_img = cv2.drawKeypoints(
    img, keyPoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("orb", show_img)
cv2.waitKey()
cv2.destroyAllWindows()
