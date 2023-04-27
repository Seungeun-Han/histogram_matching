import cv2
import numpy as np

# 입력 이미지와 목표 히스토그램 이미지 로드
src_image = cv2.imread("forest.png", cv2.IMREAD_COLOR)
dst_image = cv2.imread("sea.png", cv2.IMREAD_COLOR)
print(src_image.shape)

# R, G, B 채널에 대한 히스토그램 계산
hist_size = 256
hist_range = (0, 256)
channels = [0, 1, 2]
src_hist = cv2.calcHist([src_image], [0], None, [hist_size] * 3, hist_range)
dst_hist = cv2.calcHist([dst_image], [0], None, [hist_size] * 3, hist_range)

# 누적 분포 함수 계산
src_cdf = np.cumsum(src_hist)
dst_cdf = np.cumsum(dst_hist)

# R, G, B 채널에 대한 히스토그램 매칭
lut = np.zeros((256, 3), dtype=np.uint8)
for channel in range(3):
    for i in range(hist_size):
        diff = abs(src_cdf[i, channel] - dst_cdf[:, channel])
        j = np.argmin(diff)
        lut[i, channel] = j

# 결과 이미지 생성
result_image = cv2.LUT(src_image, lut)

# 결과 이미지 저장
cv2.imshow("result.jpg", result_image)
cv2.waitKey()
