import cv2
import numpy as np

# 입력 이미지와 목표 히스토그램 이미지 로드
srcImage = cv2.imread("C:/Users/hsyal/Desktop/etrimask_256_results_0/000004-005.png")
dstImage = cv2.imread("C:/Users/hsyal/Desktop/etri_maskDB/mask_skincolor_images/000004-005.jpg")

# 입력 이미지와 목표 히스토그램을 YCrCb 색 공간으로 변환
srcYCrCb = cv2.cvtColor(srcImage, cv2.COLOR_BGR2YCrCb)
dstYCrCb = cv2.cvtColor(dstImage, cv2.COLOR_BGR2YCrCb)

# Y 채널의 히스토그램 계산
histSize = 256
range_ = [0, 256]
histRange = range_
uniform = True
accumulate = False
srcHist = cv2.calcHist([srcYCrCb], [0], None, [histSize], histRange, uniform, accumulate)
dstHist = cv2.calcHist([dstYCrCb], [0], None, [histSize], histRange, uniform, accumulate)

# 누적 분포 함수 계산
srcCdf = srcHist.copy()
dstCdf = dstHist.copy()
for i in range(1, histSize):
    srcCdf[i] += srcCdf[i - 1]
    dstCdf[i] += dstCdf[i - 1]

# 히스토그램 매칭
lut = np.zeros((1, 256), dtype=np.uint8)
for i in range(histSize):
    minDiff = float('inf')
    index = 0
    for j in range(histSize):
        diff = abs(srcCdf[i] - dstCdf[j])
        if diff < minDiff:
            minDiff = diff
            index = j
            lut[0, i] = index

# 결과 이미지 생성
resultYCrCb = cv2.LUT(srcYCrCb[...,0], lut)

# Y 채널과 CrCb 채널을 합쳐서 결과 이미지 생성
resultChannels = [resultYCrCb, srcYCrCb[...,1], srcYCrCb[...,2]]
resultImage = cv2.merge(resultChannels)
resultImage = cv2.cvtColor(resultImage, cv2.COLOR_YCrCb2BGR)

cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)
cv2.waitKey()