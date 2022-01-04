import cv2
import numpy as np
import os


class LaneDetector:
    def __init__(self):
        self.buffer = []
        self.buffer_size = 30
        self.rho_theta_threshold = 5
        self.height_roi_ratio = (0.65, 0.85)
        self.width = None
        self.height = None
        self.total_rho_theta = None

    def find(self, img):
        # buffer size
        if len(self.buffer) >= self.buffer_size:
            self.total_rho_theta -= self.buffer[-self.buffer_size // 3]["rho_theta"]
            self.total_rho_theta -= self.buffer[-self.buffer_size // 3 * 2]["rho_theta"]
            self.total_rho_theta -= self.buffer[-self.buffer_size]["rho_theta"]
            del self.buffer[0]
        elif len(self.buffer) >= self.buffer_size // 3 * 2:
            self.total_rho_theta -= self.buffer[-self.buffer_size // 3]["rho_theta"]
            self.total_rho_theta -= self.buffer[-self.buffer_size // 3 * 2]["rho_theta"]
        elif len(self.buffer) >= self.buffer_size // 3:
            self.total_rho_theta -= self.buffer[-self.buffer_size // 3]["rho_theta"]

        # init setting
        if self.width is None and self.height is None:
            self.height, self.width = img.shape[:2]
            self.total_rho_theta = np.zeros(((self.height + self.width) * 2, 360))

        # add buffer
        self.buffer.append(
            {
                "image": img,
                "lines": None,
                "rho_theta": np.zeros(((self.height + self.width) * 2, 360)),
            }
        )

        latest_buffer = self.buffer[-1]

        # gray
        gray = cv2.cvtColor(latest_buffer["image"], cv2.COLOR_BGR2GRAY)

        # blur
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imshow(
            "blur",
            blur[
                int(self.height * self.height_roi_ratio[0]) : int(
                    self.height * self.height_roi_ratio[1]
                ),
                :,
            ],
        )

        # edge
        edge = cv2.Canny(blur, 30, 100)
        edge[: int(self.height * self.height_roi_ratio[0]), :] = 0
        edge[int(self.height * self.height_roi_ratio[1]) : self.height, :] = 0
        cv2.imshow(
            "edge",
            edge[
                int(self.height * self.height_roi_ratio[0]) : int(
                    self.height * self.height_roi_ratio[1]
                ),
                :,
            ],
        )

        # hough tf
        latest_buffer["lines"] = np.squeeze(
            cv2.HoughLinesP(edge, 4, np.pi / 36, 10, minLineLength=15, maxLineGap=5)
        )

        # convert rho theta
        if latest_buffer["lines"] is not None and np.iterable(latest_buffer["lines"]):
            for index, line in enumerate(latest_buffer["lines"]):
                if isinstance(line, np.intc):
                    continue
                x1, y1, x2, y2 = line
                rho, theta_deg, theta_rad = self.xy_line_to_rho_theta(line)

                # ignore no lanes line
                if (theta_deg >= 80 or theta_deg <= 5) and x1 <= self.width / 2:
                    continue
                if (theta_deg <= 100 or theta_deg >= 175) and x1 >= self.width / 2:
                    continue

                latest_buffer["rho_theta"][self.height + self.width + rho][
                    theta_deg
                ] += 1

        self.total_rho_theta += latest_buffer["rho_theta"] * 3

        lanes = self.find_histo_line(latest_buffer["image"])

        self.hist_draw()

        cv2.imshow("Video", latest_buffer["image"])

        return lanes

    def find_histo_line(self, img):
        segment_lane = np.zeros((4, 3))
        max_index_list = np.where(
            self.total_rho_theta.ravel() >= self.rho_theta_threshold
        )[0]

        for i, max_index in enumerate(max_index_list):
            rho = max_index // 360
            theta_deg = max_index % 360

            color = (
                self.total_rho_theta[rho][theta_deg]
                / self.total_rho_theta[max_index_list[0] // 360][
                    max_index_list[0] % 360
                ]
            ) * 255

            rho = rho - (self.height + self.width)
            theta_rad = self.deg_to_rad(theta_deg)

            # draw hist lane
            pt1, pt2 = self.rho_theta_to_xy(rho, theta_rad)
            cv2.line(img, pt1, pt2, (0, 0, color, 0), 1)

            in_vanishing_point_box, pt1_, pt2_ = cv2.clipLine(
                (
                    self.width // 2 - 30,
                    self.height // 3 * 2,
                    60,
                    30,
                ),
                pt1,
                pt2,
            )
            if not in_vanishing_point_box:
                continue

            lane = -1
            if -680 < rho and rho < -560:
                lane = 0
            if -840 < rho and rho < -720:
                lane = 1
            if -160 < rho and rho < 60:
                lane = 2
            if -400 < rho and rho < -340:
                lane = 3

            if lane != -1:
                segment_lane[lane, 0] += rho
                segment_lane[lane, 1] += theta_rad
                segment_lane[lane, 2] += 1

        lanes = []
        for total_rho, total_theta, count in segment_lane:
            if count != 0:
                rho = total_rho / count
                theta = total_theta / count
                pt1, pt2 = self.rho_theta_to_xy(rho, theta)
                cv2.line(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
                lanes.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        return lanes

    def hist_draw(self):
        total_rho_theta_equalize = (
            self.total_rho_theta >= self.rho_theta_threshold
        ).astype("uint8") * 255
        total_rho_theta_equalize = cv2.cvtColor(
            total_rho_theta_equalize, cv2.COLOR_GRAY2BGR
        )

        grid = 0
        while True:
            grid += 40
            if grid >= 4000:
                break
            color = (32, 32, 32)
            if grid < 360:
                total_rho_theta_equalize[:, grid] = color
                for n in range(4):
                    cv2.putText(
                        total_rho_theta_equalize,
                        str(grid),
                        (grid, 10 + n * 1000),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (128, 128, 128),
                    )

            if grid % 100 == 0:
                color = (64, 64, 64)
            total_rho_theta_equalize[grid, :] = color
            cv2.putText(
                total_rho_theta_equalize,
                str(grid - 2000),
                (0, grid + 5),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (128, 128, 128),
            )

        total_rho_theta_equalize_slice = cv2.hconcat(
            [
                total_rho_theta_equalize[0:1000, :, :],
                np.full((1000, 1, 3), (255, 255, 255), dtype="uint8"),
                total_rho_theta_equalize[1000:2000, :, :],
                np.full((1000, 1, 3), (255, 255, 255), dtype="uint8"),
                total_rho_theta_equalize[2000:3000, :, :],
                np.full((1000, 1, 3), (255, 255, 255), dtype="uint8"),
                total_rho_theta_equalize[3000:4000, :, :],
            ]
        )
        cv2.imshow("total_rho_theta_equalize_slice", total_rho_theta_equalize_slice)

    def xy_line_to_rho_theta(self, line):
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        # theta calc
        theta_rad = np.arctan2(dy, dx) + np.pi / 2
        theta_deg = self.rad_to_deg(theta_rad)

        # rho calc
        u_dis = np.math.hypot(dx, dy)
        u = np.array([dx, dy])
        PA = np.array([-x1, -y1])
        rho = int(np.cross(u, PA) / u_dis)

        return rho, theta_deg, theta_rad

    def rho_theta_to_xy(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        center_x = a * -rho
        center_y = b * -rho

        pt1_long = (
            int(center_x + (self.width + self.height) * (-b)),
            int(center_y + (self.width + self.height) * (a)),
        )
        pt2_long = (
            int(center_x - (self.width + self.height) * (-b)),
            int(center_y - (self.width + self.height) * (a)),
        )

        line_in_box, pt1, pt2 = cv2.clipLine(
            (
                0,
                int(self.height * self.height_roi_ratio[0]),
                self.width,
                int(
                    self.height * (self.height_roi_ratio[1] - self.height_roi_ratio[0])
                ),
            ),
            pt1_long,
            pt2_long,
        )

        if line_in_box:
            return pt1, pt2
        else:
            return pt1_long, pt2_long

    def deg_to_rad(self, deg):
        return deg / 180 * np.pi

    def rad_to_deg(self, rad):
        return int(rad / np.pi * 180)


def main():
    VIDEO_PATH = "./data/"
    det = LaneDetector()
    tm = cv2.TickMeter()
    tm.reset()
    while True:
        for i, img_file in enumerate(sorted(os.listdir(VIDEO_PATH))):
            tm.start()
            img = cv2.imread(VIDEO_PATH + img_file)

            # resize HD
            img = cv2.resize(img, (1280, 720))

            lanes = det.find(img)

            tm.stop()
            processingTime = tm.getTimeMilli()
            print("Processing Time :", processingTime, "ms")
            print(lanes)
            tm.reset()

            waitTime = int(100 - processingTime)
            if waitTime < 1:
                waitTime = 1

            key = cv2.waitKey(waitTime)

            if key == ord(" "):
                print("\n\nPressed SPACE - STOP!")
                while ord(" ") != cv2.waitKey():
                    pass
                print("Pressed SPACE - START!\n\n")
            elif key == ord("b") or key == ord("B"):
                print("\n\nPressed B - Back to start\n\n")
                break
            elif key == ord("q") or key == ord("Q"):
                break
        if key == ord("q") or key == ord("Q"):
            print("\n\nPressed Q - EXIT!\n\n")
            break


if __name__ == "__main__":
    main()
