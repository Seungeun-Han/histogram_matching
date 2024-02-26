import cv2
import numpy as np

# 입력 이미지와 목표 히스토그램 이미지 로드
srcImage = cv2.imread("./etri_maskDB_results/mask_inpainting/000001-004.jpg")
dstImage = cv2.imread("./etri_maskDB_results/skin/000001-004.jpg")

# 입력 이미지와 목표 히스토그램을 YCrCb 색 공간으로 변환
# srcYCrCb = cv2.cvtColor(srcImage, cv2.COLOR_BGR2YCrCb)
# dstYCrCb = cv2.cvtColor(dstImage, cv2.COLOR_BGR2YCrCb)

# split
img_b, img_g, img_r = cv2.split(srcImage)
target_b, target_g, target_r = cv2.split(dstImage)


# Y 채널의 히스토그램 계산
histSize = 256
range_ = [0, 256]
histRange = range_
uniform = True
accumulate = False
srcHist = cv2.calcHist([img_b], [0], None, [histSize], histRange, uniform, accumulate)
dstHist = cv2.calcHist([target_b], [0], None, [histSize], histRange, uniform, accumulate)

g_srcHist = cv2.calcHist([img_g], [0], None, [histSize], histRange, uniform, accumulate)
g_dstHist = cv2.calcHist([target_g], [0], None, [histSize], histRange, uniform, accumulate)

r_srcHist = cv2.calcHist([img_r], [0], None, [histSize], histRange, uniform, accumulate)
r_dstHist = cv2.calcHist([target_r], [0], None, [histSize], histRange, uniform, accumulate)


print(srcHist[0])
# # 배경 제거
srcHist[0] = [0]
dstHist[0] = [0]

g_srcHist[0] = [0]
g_dstHist[0] = [0]

r_srcHist[0] = [0]
r_dstHist[0] = [0]
print(srcHist[0])


srcCdf_norm = cv2.normalize(srcHist, None, 0, 1, cv2.NORM_MINMAX)
dstCdf_norm = cv2.normalize(dstHist, None, 0, 1, cv2.NORM_MINMAX)

g_srcCdf_norm = cv2.normalize(g_srcHist, None, 0, 1, cv2.NORM_MINMAX)
g_dstCdf_norm = cv2.normalize(g_dstHist, None, 0, 1, cv2.NORM_MINMAX)

r_srcCdf_norm = cv2.normalize(r_srcHist, None, 0, 1, cv2.NORM_MINMAX)
r_dstCdf_norm = cv2.normalize(r_dstHist, None, 0, 1, cv2.NORM_MINMAX)

# # 배경 제거
# srcHist[0] = [0]
# dstHist[0] = [0]

# 누적 분포 함수 계산
srcCdf = srcHist.copy()
dstCdf = dstHist.copy()
for i in range(1, histSize):
    srcCdf[i] += srcCdf[i - 1]
    dstCdf[i] += dstCdf[i - 1]

srcCdf_norm = cv2.normalize(srcCdf, None, 0, 1, cv2.NORM_MINMAX)
dstCdf_norm = cv2.normalize(dstCdf, None, 0, 1, cv2.NORM_MINMAX)

g_srcCdf = g_srcHist.copy()
g_dstCdf = g_dstHist.copy()
for i in range(1, histSize):
    g_srcCdf[i] += g_srcCdf[i - 1]
    g_dstCdf[i] += g_dstCdf[i - 1]

g_srcCdf_norm = cv2.normalize(g_srcCdf, None, 0, 1, cv2.NORM_MINMAX)
g_dstCdf_norm = cv2.normalize(g_dstCdf, None, 0, 1, cv2.NORM_MINMAX)

r_srcCdf = r_srcHist.copy()
r_dstCdf = r_dstHist.copy()
for i in range(1, histSize):
    r_srcCdf[i] += r_srcCdf[i - 1]
    r_dstCdf[i] += r_dstCdf[i - 1]

r_srcCdf_norm = cv2.normalize(r_srcCdf, None, 0, 1, cv2.NORM_MINMAX)
r_dstCdf_norm = cv2.normalize(r_dstCdf, None, 0, 1, cv2.NORM_MINMAX)


# 히스토그램 매칭
lut = np.zeros((1, 256), dtype=np.uint8)
for i in range(histSize):
    minDiff = float('inf')
    index = 0
    for j in range(histSize):
        diff = abs(srcCdf_norm[i] - dstCdf_norm[j])
        if diff < minDiff:
            minDiff = diff
            index = j
            lut[0, i] = index

g_lut = np.zeros((1, 256), dtype=np.uint8)
for i in range(histSize):
    minDiff = float('inf')
    index = 0
    for j in range(histSize):
        diff = abs(g_srcCdf_norm[i] - g_dstCdf_norm[j])
        if diff < minDiff:
            minDiff = diff
            index = j
            g_lut[0, i] = index

r_lut = np.zeros((1, 256), dtype=np.uint8)
for i in range(histSize):
    minDiff = float('inf')
    index = 0
    for j in range(histSize):
        diff = abs(r_srcCdf_norm[i] - r_dstCdf_norm[j])
        if diff < minDiff:
            minDiff = diff
            index = j
            r_lut[0, i] = index


# 결과 이미지 생성
resultYCrCb = cv2.LUT(srcImage[...,0], lut)
# 결과 이미지 생성
g_result = cv2.LUT(srcImage[...,1], g_lut)
# 결과 이미지 생성
r_result = cv2.LUT(srcImage[...,2], r_lut)

# Y 채널과 CrCb 채널을 합쳐서 결과 이미지 생성
resultChannels = [resultYCrCb, g_result, r_result]
resultImage = cv2.merge(resultChannels)
# resultImage = cv2.cvtColor(resultImage, cv2.COLOR_YCrCb2BGR)

cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)
cv2.waitKey()