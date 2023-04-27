import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

# sorting하는 방법 : 승은 작성
def histogram_matching_by_sorting(srcImage, srcHist, dstHist):
    # srcHist = (255 * srcHist / max(srcHist)).astype(np.uint8)
    # dstHist = (255 * dstHist / max(dstHist)).astype(np.uint8)

    # 이중 리스트를 1차원 리스트로
    # print(srcHist)
    srcHist = list(itertools.chain(*srcHist))
    dstHist = list(itertools.chain(*dstHist))
    # print(srcHist)


    # sorted_srcHist = sorted(srcHist, reverse=True)
    # sorted_dstHist = sorted(dstHist, reverse=True)
    # print(sorted_srcHist)
    # print(sorted_dstHist)

    sorted_index_srcHist = np.argsort(srcHist)[::-1]
    sorted_index_dstHist = np.argsort(dstHist)[::-1]
    # print(sorted_index_srcHist)


    lut = np.zeros((256), dtype=np.uint8)
    for i in range(histSize):
        # src_index = np.where(srcHist == sorted_srcHist[i])[0]
        # dst_index = np.where(dstHist == sorted_dstHist[i])[0]
        lut[sorted_index_srcHist[i]] = sorted_index_dstHist[i]

    resultImage = cv2.LUT(srcImage, lut)

    return resultImage

def histogram_matching_fromGPT(srcImage, srcHist, dstHist):
    srcCdf = np.cumsum(srcHist)
    dstCdf = np.cumsum(dstHist)

    lut = np.zeros((256), dtype=np.uint8)
    for i in range(256):
        minDiff = float('inf')
        index = 0
        for j in range(256):
            diff = abs(srcCdf[i] - dstCdf[j])
            if diff < minDiff:
                minDiff = diff
                index = j
                lut[i] = index

    resultImage = cv2.LUT(srcImage, lut)

    return resultImage


srcImage = cv2.imread("gray1.png")
dstImage = cv2.imread("gray2.png")


srcImage_ravel = srcImage.ravel()
dstImage_ravel = dstImage.ravel()

histSize = 256
range_ = [0, 256]
histRange = range_
uniform = True
accumulate = False
srcHist = cv2.calcHist([srcImage], [0], None, [histSize], histRange, uniform, accumulate)
dstHist = cv2.calcHist([dstImage], [0], None, [histSize], histRange, uniform, accumulate)
# srcHist, srccounts = np.histogram(srcImage, 256, [0,256])
# dstHist, dstcounts = np.histogram(dstImage, 256, [0,256])


# 승은 sorting 방식
resultImage = histogram_matching_by_sorting(srcImage, srcHist, dstHist)

# GPT 방식
# resultImage = histogram_matching_fromGPT(srcImage, srcHist, dstHist)

resultHist = cv2.calcHist([resultImage], [0], None, [histSize], histRange, uniform, accumulate)

# cv2.imwrite("gray_test_result.png", resultImage)

cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)

plt.figure("result", figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(srcHist)
plt.subplot(1, 3, 2)
plt.plot(dstHist)
plt.subplot(1, 3, 3)
plt.plot(resultHist)
plt.show()

cv2.waitKey()