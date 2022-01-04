"""
4. Optical Flow
- stitching.zip에서 dog_a, dog_b 두 사진을 이용해서 Good Feature to Tracking을 추출하고 Pyramid Lucas-Kanade 알고리즘을 적용해서 Optical Flow를 구하는 코드를 작성하시오.
- stitching.zip에서 dog_a, dog_b 두 사진을 이용해서 Farneback과 DualTVL1 Optical Flow 알고리즘을 구하는 코드를 작성하시오.
"""
import cv2
import numpy as np

imgs = []
imgs.append(cv2.imread("./stitching/dog_a.jpg"))
imgs.append(cv2.imread("./stitching/dog_b.jpg"))

prev_pts = None
prev_gray_frame = None
tracks = None

result_imgs = []
for img in imgs:
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        pts, status, errors = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame,
            gray_frame,
            prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=5,
            criteria=(cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        good_pts = pts[status == 1]
        if tracks is None:
            tracks = good_pts
        else:
            tracks = np.vstack((tracks, good_pts))
        copy_img = np.copy(img)
        for p in tracks:
            cv2.circle(copy_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        result_imgs.append(copy_img)
    else:
        pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10)
        pts = pts.reshape(-1, 1, 2)
    prev_pts = pts
    prev_gray_frame = gray_frame


for index, result_img in enumerate(result_imgs):
    cv2.imwrite(f"./4_1_{index+1}.png", result_img)
    cv2.imshow("img", result_img)
    cv2.waitKey()

cv2.destroyAllWindows()


# Farneback과 DualTVL1 Optical Flow 알고리즘
def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(
                img, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4
            )
    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)
    norm_opt_flow = (cv2.cvtColor(norm_opt_flow, cv2.COLOR_GRAY2BGR) * 255).astype(
        np.uint8
    )
    # print(img.shape, norm_opt_flow.shape)
    result = cv2.vconcat([img, norm_opt_flow])
    # cv2.imshow("optical flow", result)
    # k = cv2.waitKey(0)
    return result


result_imgs = []

# Farneback
prev_frame = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
opt_flow = cv2.calcOpticalFlowFarneback(
    prev_frame,
    gray,
    None,
    0.5,
    5,
    13,
    10,
    5,
    1.1,
    cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)
result_imgs.append(display_flow(np.copy(imgs[1]), opt_flow))

# DualTVL1
flow_DualTVL1 = cv2.createOptFlow_DualTVL1()
opt_flow = flow_DualTVL1.calc(prev_frame, gray, None)
flow_DualTVL1.setUseInitialFlow(True)

result_imgs.append(display_flow(np.copy(imgs[1]), opt_flow))


for index, result_img in enumerate(result_imgs):
    cv2.imwrite(f"./4_2_{index+1}.png", result_img)
    cv2.imshow("img", result_img)
    cv2.waitKey()

cv2.destroyAllWindows()
