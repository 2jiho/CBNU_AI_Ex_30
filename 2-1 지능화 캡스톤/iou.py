def iou(bbox1: list, bbox2: list) -> float:
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    # 바운딩박스 넓이
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # 교집합
    intersection_x1 = max(cx1 - (w1 / 2), cx2 - (w2 / 2))
    intersection_y1 = max(cy1 - (h1 / 2), cy2 - (h2 / 2))
    intersection_x2 = min(cx1 + (w1 / 2), cx2 + (w2 / 2))
    intersection_y2 = min(cy1 + (h1 / 2), cy2 + (h2 / 2))
    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)

    # 교집합, 합집합 넓이
    intersection_area = intersection_w * intersection_h
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area


if __name__ == "__main__":

    print(iou([0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]))
    print(iou([0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]))
    print(iou([0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.05, 0.05]))
