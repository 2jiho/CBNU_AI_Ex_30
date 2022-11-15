from iou import iou


def ap(det: list, gt: list, iou_th=0.5) -> float:
    """
    input:
        det = [[bbox, conf], ...]
        gt = [bbox, ...]
        bbox = [cx, cy, w, h]
    output:
        ap_score
    """

    # det을 conf기준 내림차순 정렬
    det = sorted(det, key=lambda x: -x[1])

    # TP 체크
    tp = []
    for det_object in det:
        det_bbox = det_object[0]
        max_iou = 0.0
        for gt_bbox in gt:
            max_iou = max(max_iou, iou(det_bbox, gt_bbox))
        if max_iou > iou_th:
            tp.append(True)
        else:
            tp.append(False)

    # pr곡선 그리기
    pr_curve = []
    count_tp = 0  # 누적TP
    for index, det_object in enumerate(det):
        if tp[index]:
            count_tp += 1

        precision = count_tp / (index + 1)
        recall = count_tp / len(gt)
        pr_curve.append([recall, precision])
    # pr곡선 보정
    for index in reversed(range(1, len(pr_curve))):
        if pr_curve[index - 1][1] < pr_curve[index][1]:
            pr_curve[index - 1][1] = pr_curve[index][1]

    # pr 넓이 구하기
    area = pr_curve[0][0] * pr_curve[0][1]
    for index in range(1, len(pr_curve)):
        diff_recall = pr_curve[index][0] - pr_curve[index - 1][0]
        precision = pr_curve[index][1]
        area += diff_recall * precision

    return area


if __name__ == "__main__":
    det = [
        [[0.5, 0.5, 0.1, 0.1], 0.95],
        [[0.5, 0.5, 0.1, 0.1], 0.91],
        [[0.5, 0.5, 0.1, 0.1], 0.85],
        [[0.5, 0.5, 0.1, 0.1], 0.81],
        [[0.5, 0.5, 0.1, 0.1], 0.78],
        [[0.1, 0.1, 0.1, 0.1], 0.68],
        [[0.5, 0.5, 0.1, 0.1], 0.57],
        [[0.5, 0.5, 0.1, 0.1], 0.45],
        [[0.1, 0.1, 0.1, 0.1], 0.43],
        [[0.1, 0.1, 0.1, 0.1], 0.13],
    ]
    gt = [
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1, 0.1],
    ]

    print(ap(det, gt))
        