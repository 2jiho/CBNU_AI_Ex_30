import cv2

print(f"opencv version: {cv2.__version__}")

img_path = "lena.png"


img = cv2.imread(img_path)
print(f"org shape: {img.shape}")
cv2.imshow("img", img)
cv2.waitKey(0)

w, h = 128, 256
resized_img = cv2.resize(img, (w, h))
print(f"resized shape: {resized_img.shape}")
cv2.imshow("img", resized_img)
cv2.waitKey(0)

w_mult, h_mult = 0.5, 0.75
resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult)
print(f"img shape: {resized_img.shape}")
cv2.imshow("img", resized_img)
cv2.waitKey(0)

w_mult, h_mult = 2, 4
resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult, cv2.INTER_NEAREST)
print(f"img shape: {resized_img.shape}")
cv2.imshow("img", resized_img)
cv2.waitKey(0)

img_filp_along_x = cv2.flip(img, 0)
img_filp_along_x_along_y = cv2.flip(img_filp_along_x, 1)
img_filpped_xy = cv2.flip(img, -1)
assert img_filpped_xy.all() == img_filp_along_x_along_y.all()
cv2.imshow("img", img_filpped_xy)
cv2.waitKey(0)
