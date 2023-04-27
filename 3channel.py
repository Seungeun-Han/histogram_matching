import cv2
import numpy as np
import matplotlib.pyplot as plt

# 입력 이미지와 목표 히스토그램 이미지 로드
srcImage = cv2.imread("forest.png")
dstImage = cv2.imread("sea.png")
# srcImage = cv2.imread("./etri_maskDB_results/mask_inpainting/000006-005.jpg") #   forest.png
# dstImage = cv2.imread("./etri_maskDB_results/skin/000006-005.jpg")

# split
img_b, img_g, img_r = cv2.split(srcImage)
target_b, target_g, target_r = cv2.split(dstImage)

original_b = img_b.ravel()
original_g = img_g.ravel()
original_r = img_r.ravel()



zero_index = np.where((original_b==0)&(original_g==0)&(original_r==0))
not_zero_index = np.where((original_b!=0)&(original_g!=0)&(original_r!=0))
print(len(zero_index[0]))

specified_b = target_b.ravel()
specified_g = target_g.ravel()
specified_r = target_r.ravel()

target_zero_index = np.where((specified_b==0)&(specified_g==0)&(specified_r==0))
not_target_zero_index = np.where((original_b!=0)&(original_g!=0)&(original_r!=0))
print(len(target_zero_index[0]))

print(len(original_b[not_zero_index]))


# R 채널의 히스토그램 계산
# srcHist, _ = np.histogram(original_b[not_zero_index], 256, [0,256])
# dstHist, _ = np.histogram(specified_b[not_target_zero_index], 256, [0,256])
srcHist, _ = np.histogram(srcImage, 256, [0,256])
dstHist, _ = np.histogram(dstImage, 256, [0,256])
print(srcHist.shape)
# print(dstHist)

# srcHist[0] -= len(zero_index[0])
# dstHist[0] -= len(target_zero_index[0])
srcHist[:5] = 0
dstHist[:5] = 0

print(srcHist)
srcHist = (255 * srcHist / max(srcHist)).astype(np.uint8)
dstHist = (255 * dstHist / max(dstHist)).astype(np.uint8)
print(srcHist)

# 누적 분포 함수 계산
srcCdf = srcHist.cumsum()
dstCdf = dstHist.cumsum()


# 히스토그램 매칭
lut = np.zeros((1, 256), dtype=np.uint8)
for i in range(256):
    minDiff = float('inf')
    index = 0
    for j in range(256):
        # diff = abs(srcCdf_norm[i] - dstCdf_norm[j])
        diff = abs(srcCdf[i] - dstCdf[j])
        # print(i, j, diff)
        if diff < minDiff:
            minDiff = diff
            index = j
            lut[0, i] = index



# 결과 이미지 생성
# resultG = cv2.LUT(srcImage, lut)
# resultB = cv2.LUT(srcImage, lut)
# resultR = cv2.LUT(srcImage, lut)
# resultYCrCb = cv2.LUT(original_b[not_zero_index], lut)
# 결과 이미지 생성
# g_result = cv2.LUT(original_g[not_zero_index], lut)
# 결과 이미지 생성
# r_result = cv2.LUT(original_r[not_zero_index], lut)

# resultChannels = [resultYCrCb, g_result, r_result]
# resultImage = cv2.merge(resultChannels)


# plt.figure("result")
# plt.subplot(1, 3, 1)
# plt.hist(resultYCrCb, 256, [1, 256])
# plt.subplot(1, 3, 2)
# plt.hist(g_result, 256, [1, 256])
# plt.subplot(1, 3, 3)
# plt.hist(r_result, 256, [1, 256])
# plt.show()

cv2.imshow("input", srcImage)
cv2.imshow("target", dstImage)
cv2.imshow("result", resultG)

cv2.waitKey()