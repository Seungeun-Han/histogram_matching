import cv2

image_path = "./etri_maskDB_results/skin/000001-004.jpg"

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

cv2.imwrite("skin.png", image)