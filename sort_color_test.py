import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

# sorting하는 방법 : 승은 작성
def histogram_matching_by_sorting(srcImage, dstImage, srcHist, dstHist):
    # srcHist = (255 * (srcHist / max(srcHist))).astype(np.uint8)
    # dstHist = (255 * (dstHist / max(dstHist))).astype(np.uint8)

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

    resultImage = srcImage.copy()
    lut = np.zeros((256), dtype=np.uint8)
    for i in range(histSize):
        # src_index = np.where(srcHist == sorted_srcHist[i])[0]
        # dst_index = np.where(dstHist == sorted_dstHist[i])[0]
        lut[sorted_index_srcHist[i]] = sorted_index_dstHist[i]

        # 픽셀 값 3개 다 가져오기
        # src_value_list = np.where(srcImage[:,:,0] == sorted_index_srcHist[i])
        # dst_value_list = np.where(dstImage[:,:,0] == sorted_index_dstHist[i])
        # print(i, sorted_index_srcHist[i], sorted_index_dstHist[i], len(src_value_list[0]), len(dst_value_list[0]))
        # # print(srcImage[src_value_list])
        # # print(dstImage[dst_value_list])
        # # print(len(src_value_list[0]), len(src_value_list[1]))
        # if len(dst_value_list[0]) != 0:
        #     for x, y in zip(src_value_list[0], src_value_list[1]):
        #         resultImage[x][y][0] = dstImage[dst_value_list][0][0]

    resultImage = cv2.LUT(srcImage, lut)

    return resultImage
    # return lut

def histogram_matching_fromGPT(srcImage, srcHist, dstHist):
    srcHist = (255 * (srcHist / max(srcHist))).astype(np.uint8)
    dstHist = (255 * (dstHist / max(dstHist))).astype(np.uint8)
    # r_srcCdf_norm = cv2.normalize(r_srcCdf, None, 0, 1, cv2.NORM_MINMAX)
    # r_dstCdf_norm = cv2.normalize(r_dstCdf, None, 0, 1, cv2.NORM_MINMAX)

    srcCdf = np.cumsum(srcHist)
    dstCdf = np.cumsum(dstHist)

    srcCdf = srcCdf * dstHist.max() / srcCdf.max()
    dstCdf = dstCdf * srcHist.max() / dstCdf.max()

    lut = np.zeros((256), dtype=np.uint8)
    for i in range(256):
        minDiff = float('inf')
        for j in range(256):
            diff = abs(float(srcCdf[i] - dstCdf[j]))
            if diff < minDiff:
                minDiff = diff
                lut[i] = j

    resultImage = cv2.LUT(srcImage, lut)

    return resultImage


# srcImage = cv2.imread("gray1.png")
# dstImage = cv2.imread("gray2.png")
# srcImage = cv2.imread("forest.png")
# dstImage = cv2.imread("sea.png")
srcImage = cv2.imread("./etri_maskDB_results/mask_inpainting/000001-004.jpg")
dstImage = cv2.imread("./etri_maskDB_results/skin/000001-004.jpg")

# split
img_b, img_g, img_r = cv2.split(srcImage)
target_b, target_g, target_r = cv2.split(dstImage)

original_b = img_b.ravel()
original_g = img_g.ravel()
original_r = img_r.ravel()
zero_index = np.where((original_b==0)&(original_g==0)&(original_r==0))
not_zero_index = np.where((original_b==0)|(original_g==0)|(original_r==0))

specified_b = target_b.ravel()
specified_g = target_g.ravel()
specified_r = target_r.ravel()
target_zero_index = np.where((specified_b==0)&(specified_g==0)&(specified_r==0))
not_target_zero_index = np.where((specified_b==0)|(specified_g==0)|(specified_r==0))


# srcImage_ravel = srcImage.ravel()
# dstImage_ravel = dstImage.ravel()

histSize = 256
range_ = [0, 256]
histRange = range_
uniform = True
accumulate = False
srcHist = cv2.calcHist([srcImage], [0], None, [histSize], histRange, uniform, accumulate)
dstHist = cv2.calcHist([dstImage], [0], None, [histSize], histRange, uniform, accumulate)
g_srcHist = cv2.calcHist([srcImage], [1], None, [histSize], histRange, uniform, accumulate)
g_dstHist = cv2.calcHist([dstImage], [1], None, [histSize], histRange, uniform, accumulate)
r_srcHist = cv2.calcHist([srcImage], [2], None, [histSize], histRange, uniform, accumulate)
r_dstHist = cv2.calcHist([dstImage], [2], None, [histSize], histRange, uniform, accumulate)

# 0,0,0 제거
# srcHist[0] -= len(zero_index[0])
# dstHist[0] -= len(target_zero_index[0])
# g_srcHist[0] -= len(zero_index[0])
# g_dstHist[0] -= len(target_zero_index[0])
# r_srcHist[0] -= len(zero_index[0])
# r_dstHist[0] -= len(target_zero_index[0])
srcHist[:20] = 0
dstHist[:20] = 0
g_srcHist[:20] = 0
g_dstHist[:20] = 0
r_srcHist[:20] = 0
r_dstHist[:20] = 0

# srcHist[0] += len(not_zero_index[0])
# dstHist[0] += len(not_target_zero_index[0])
# g_srcHist[0] += len(not_zero_index[0])
# g_dstHist[0] += len(not_target_zero_index[0])
# r_srcHist[0] += len(not_zero_index[0])
# r_dstHist[0] += len(not_target_zero_index[0])


# 승은 sorting 방식
# b_resultImage = histogram_matching_by_sorting(srcImage[:,:,0], dstImage, srcHist, dstHist)
# g_resultImage = histogram_matching_by_sorting(srcImage[:,:,1], dstImage, g_srcHist, g_dstHist)
# r_resultImage = histogram_matching_by_sorting(srcImage[:,:,2], dstImage, r_srcHist, r_dstHist)

# resultImage = histogram_matching_by_sorting(srcImage, dstImage, srcHist, dstHist)

# b 채널에 대해서 계산된 lut 사용 방식
# lut = histogram_matching_by_sorting(srcImage, dstImage, srcHist, dstHist)
# b_resultImage = cv2.LUT(srcImage[:,:,0], lut)
# g_resultImage = cv2.LUT(srcImage[:,:,1], lut)
# r_resultImage = cv2.LUT(srcImage[:,:,2], lut)

# GPT 방식
b_resultImage = histogram_matching_fromGPT(srcImage[:,:,0], srcHist, dstHist)
g_resultImage = histogram_matching_fromGPT(srcImage[:,:,1], g_srcHist, g_dstHist)
r_resultImage = histogram_matching_fromGPT(srcImage[:,:,2], r_srcHist, r_dstHist)

# merge
resultChannels = [b_resultImage, g_resultImage, r_resultImage]
resultImage = cv2.merge(resultChannels)
#
# calc result hist
b_resultHist = cv2.calcHist([b_resultImage], [0], None, [histSize], histRange, uniform, accumulate)
g_resultHist = cv2.calcHist([g_resultImage], [0], None, [histSize], histRange, uniform, accumulate)
r_resultHist = cv2.calcHist([r_resultImage], [0], None, [histSize], histRange, uniform, accumulate)

b_resultHist[0] -= len(zero_index[0])
g_resultHist[0] -= len(zero_index[0])
r_resultHist[0] -= len(zero_index[0])

# cv2.imwrite("gray_test_result.png", resultImage)

cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)

plt.figure("b_result", figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(srcHist)
plt.subplot(1, 3, 2)
plt.plot(dstHist)
plt.subplot(1, 3, 3)
plt.plot(b_resultHist)

plt.figure("g_result", figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(g_srcHist)
plt.subplot(1, 3, 2)
plt.plot(g_dstHist)
plt.subplot(1, 3, 3)
plt.plot(g_resultHist)

plt.figure("r_result", figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(r_srcHist)
plt.subplot(1, 3, 2)
plt.plot(r_dstHist)
plt.subplot(1, 3, 3)
plt.plot(r_resultHist)
plt.show()

cv2.waitKey()