"""
2. Matching
- stitching.zip에서 각 영상셋(boat, budapest, newpaper, s1~s2)에서 두 장을 선택하고 각 영상에서 각각 SIFT, SURF, ORB를 추출한 후에 매칭 및 RANSAC을 통해서 두 장의 영상간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping 하는 코드를 작성하시오.
"""
import cv2
import numpy as np

imgs_list = []
imgs_list.append(
    [
        cv2.imread("./stitching/boat1.jpg"),
        cv2.imread("./stitching/boat2.jpg"),
    ]
)
imgs_list.append(
    [
        cv2.imread("./stitching/budapest1.jpg"),
        cv2.imread("./stitching/budapest2.jpg"),
    ]
)
imgs_list.append(
    [
        cv2.imread("./stitching/newspaper1.jpg"),
        cv2.imread("./stitching/newspaper2.jpg"),
    ]
)
imgs_list.append(
    [
        cv2.imread("./stitching/s1.jpg"),
        cv2.imread("./stitching/s2.jpg"),
    ]
)

result_imgs = []
for imgs in imgs_list:
    # img0 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    # img1 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
    img0 = imgs[0]
    img1 = imgs[1]
    result = []
    # surf = cv2.xfeatures2d.SURF_create(10000)
    # surf.setExtended(True)
    # surf.setNOcatves(3)
    # surf.setNOcatvesLayers(10)
    # surf.setUpright(False)
    # kps0, fea0 = surf.detectAndCompute(img0, None)
    # kps1, fea1 = surf.detectAndCompute(img1, None)
    # matcher = cv2.BFMatcher_create(cv2.NORM_L1, False)
    # matches = matcher.match(fea0, fea1)
    # pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    # pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    # H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    # surf_img = cv2.drawMatches(
    #     img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None
    # )
    # cv2.putText(
    #     surf_img, "SURF", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 0), 5
    # )
    # result.append(surf_img)

    sift = cv2.xfeatures2d.SIFT_create(50)
    kps0, fea0 = sift.detectAndCompute(img0, None)
    kps1, fea1 = sift.detectAndCompute(img1, None)
    matcher = cv2.BFMatcher_create(cv2.NORM_L1, False)
    matches = matcher.match(fea0, fea1)
    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    sift_img = cv2.drawMatches(
        img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None
    )
    cv2.putText(
        sift_img, "SIFT", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 0), 5
    )
    result.append(sift_img)

    orb = cv2.ORB_create(100)
    kps0, fea0 = orb.detectAndCompute(img0, None)
    kps1, fea1 = orb.detectAndCompute(img1, None)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
    matches = matcher.match(fea0, fea1)
    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    orb_img = cv2.drawMatches(
        img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None
    )
    cv2.putText(
        orb_img, "ORB", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 0), 5
    )
    result.append(orb_img)

    result_imgs.append(result)


for index, result_img in enumerate(result_imgs):
    result = cv2.vconcat(result_img)
    cv2.imwrite(f"./2_{index+1}.png", result)
    cv2.imshow("img", result)
    cv2.waitKey()

cv2.destroyAllWindows()
