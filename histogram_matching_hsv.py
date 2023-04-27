"""import cv2
import numpy as np

# input 이미지와 target 이미지 로드
# srcImage = cv2.imread("gray1.png")
# dstImage = cv2.imread("gray2.png")
srcImage = cv2.imread("forest.png")
dstImage = cv2.imread("sea.png")
# srcImage = cv2.imread("./etri_maskDB_results/mask_inpainting/000001-004.jpg")
# dstImage = cv2.imread("./etri_maskDB_results/skin/000001-004.jpg")

# input 이미지와 target 이미지를 HSV 색 공간으로 변환
hsv_input = cv2.cvtColor(srcImage, cv2.COLOR_BGR2HSV)
hsv_target = cv2.cvtColor(dstImage, cv2.COLOR_BGR2HSV)

# input 이미지의 V 채널 히스토그램 계산
hist_input, bins = np.histogram(hsv_input[:, :, 2].flatten(), 256, [0, 256])

# target 이미지의 V 채널 히스토그램 계산
hist_target, bins = np.histogram(hsv_target[:, :, 2].flatten(), 256, [0, 256])

# 누적 히스토그램 계산
cumsum_input = hist_input.cumsum()
cumsum_target = hist_target.cumsum()

# 누적 히스토그램 정규화
cumsum_input_normalized = cumsum_input * hist_target.max() / cumsum_input.max()
cumsum_target_normalized = cumsum_target * hist_input.max() / cumsum_target.max()

# 룩업 테이블 생성
lut = np.zeros((256, 1), dtype=np.uint8)

# 룩업 테이블 생성
for i in range(256):
    j = 255
    while True:
        if cumsum_input_normalized[i] <= cumsum_target_normalized[j]:
            lut[i] = j
            break
        j -= 1

# 히스토그램 매칭
hsv_result = hsv_input.copy()

hsv_result[:, :, 2] = cv2.LUT(hsv_input[:, :, 2], lut)

# 결과 이미지를 BGR 색 공간으로 변환
resultImage = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

# 결과 이미지 저장
# cv2.imwrite("result_image.png", result)
cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)

cv2.waitKey()
"""

"""import cv2
import numpy as np

# 입력 이미지와 목표 이미지 로드
input_image = cv2.imread("forest.png", cv2.IMREAD_COLOR)
target_image = cv2.imread("sea.png", cv2.IMREAD_COLOR)
# input_image = cv2.imread("./etri_maskDB_results/mask_inpainting/000001-004.jpg")
# target_image = cv2.imread("./etri_maskDB_results/skin/000001-004.jpg")


# 입력 이미지와 목표 이미지를 HSV로 변환
input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# 입력 이미지와 목표 이미지에서 블루 채널만 추출
input_blue_channel = input_hsv[:, :, 2]
target_blue_channel = target_hsv[:, :, 2]

# 히스토그램 매칭을 위한 LUT 생성
hist, bins = np.histogram(input_blue_channel.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
lut = np.interp(target_blue_channel.flatten(), bins[:-1], cdf_normalized)

# LUT를 이용하여 목표 이미지의 블루 채널 히스토그램 매칭
target_blue_channel_matched = lut.reshape((target_image.shape[0], target_image.shape[1])).astype(np.uint8)

# 목표 이미지의 블루 채널을 매칭된 이미지로 교체
target_hsv[:, :, 2] = target_blue_channel_matched

# 매칭된 이미지를 다시 BGR로 변환
matched_image = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)

# 결과 이미지 출력
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
