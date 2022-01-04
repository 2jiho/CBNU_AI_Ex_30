import cv2

print(f"opencv version: {cv2.__version__}")

mouse_pressed = False
s_x = s_y = e_x = e_y = -1

img_path = "lena.png"
img = cv2.imread(img_path)
show_img = img.copy()


def mouse_callback(event, x, y, flags, param):
    global img, show_img, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        e_x, e_y = x, y

    if event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
        e_x, e_y = x, y

    if event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        img = show_img.copy()
        s_x = s_y = e_x = e_y = -1


if __name__ == "__main__":
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_callback)
    mode = "r"
    cv2.setWindowTitle("img", "rectangle mode")
    while True:
        show_img = img.copy()
        if mode == "r":
            cv2.rectangle(show_img, (s_x, s_y), (e_x, e_y), (0, 255, 0), thickness=3)
        elif mode == "l":
            cv2.line(show_img, (s_x, s_y), (e_x, e_y), (0, 255, 0), thickness=3)
        elif mode == "a":
            cv2.arrowedLine(show_img, (s_x, s_y), (e_x, e_y), (0, 255, 0), thickness=3)

        cv2.imshow("img", show_img)
        key = cv2.waitKey(33)
        if key == 27:
            break
        elif key == ord("r"):
            # 사각
            mode = "r"
            cv2.setWindowTitle("img", "rectangle mode")
        elif key == ord("l"):
            # 선
            mode = "l"
            cv2.setWindowTitle("img", "line mode")
        elif key == ord("a"):
            # 화살표
            mode = "a"
            cv2.setWindowTitle("img", "arrowedLine mode")
        elif key == ord("w"):
            # 저장
            cv2.imwrite("lena_draw.png", img)
        elif key == ord("c"):
            # clear
            img = cv2.imread(img_path)
