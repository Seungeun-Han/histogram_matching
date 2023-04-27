import cv2
import numpy as np
import matplotlib.pyplot as plt

# 입력 이미지와 목표 히스토그램 이미지 로드
# srcImage = cv2.imread("forest.png")
# dstImage = cv2.imread("sea.png")
srcImage = cv2.imread("./etri_maskDB_results/mask_inpainting/000006-005.jpg") #   forest.png
dstImage = cv2.imread("./etri_maskDB_results/skin/000006-005.jpg")  #   sea.png
# srcImage = cv2.imread("inpainting.png")
# dstImage = cv2.imread("skin.png")

# 입력 이미지와 목표 히스토그램을 YCrCb 색 공간으로 변환
# srcYCrCb = cv2.cvtColor(srcImage, cv2.COLOR_BGR2YCrCb)
# dstYCrCb = cv2.cvtColor(dstImage, cv2.COLOR_BGR2YCrCb)

# split
img_b, img_g, img_r = cv2.split(srcImage)
target_b, target_g, target_r = cv2.split(dstImage)

original_b = img_b.ravel()
original_g = img_g.ravel()
original_r = img_r.ravel()

# print(len(original_b[np.where(original_b==1)]))

zero_index = np.where((original_b==0)&(original_g==0)&(original_r==0))
not_zero_index = np.where((original_b!=0)&(original_g!=0)&(original_r!=0))
# print(len(zero_index[0]))

print(zero_index[0])

specified_b = target_b.ravel()
specified_g = target_g.ravel()
specified_r = target_r.ravel()

target_zero_index = np.where((specified_b==0)&(specified_g==0)&(specified_r==0))
# print(len(target_zero_index[0]))

# 3 채널의 히스토그램 계산
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

# print(list(map(srcHist, int)))
print(srcHist[0])
# # # 배경 제거
srcHist[0] -= len(zero_index[0])
dstHist[0] -= len(target_zero_index[0])

g_srcHist[0] -= len(zero_index[0])
g_dstHist[0] -= len(target_zero_index[0])

r_srcHist[0] -= len(zero_index[0])
r_dstHist[0] -= len(target_zero_index[0])
# srcHist[:10] = 0
# dstHist[:10] = 0
#
# g_srcHist[:10] = 0
# g_dstHist[:10] = 0
#
# r_srcHist[:10] = 0
# r_dstHist[:10] = 0
print(srcHist[0])
print(dstHist[0])

print("srcHist[1]", srcHist[1])
print("dstHist[1]", dstHist[1])

if sum(dstHist) > sum(srcHist):
    rate = sum(dstHist) / sum(srcHist)
    srcHist *= rate
    g_srcHist *= rate
    r_srcHist *= rate
else:
    rate = sum(srcHist) / sum(dstHist)
    dstHist *= rate
    g_dstHist *= rate
    r_dstHist *= rate

print("sum(r_srcHist)", sum(r_srcHist))
print("sum(r_dstHist)", sum(r_dstHist))

# srcHist_norm = cv2.normalize(srcHist, None, 0, 256, cv2.NORM_MINMAX)
# dstHist_norm = cv2.normalize(dstHist, None, 0, 256, cv2.NORM_MINMAX)
#
# g_srcHist_norm = cv2.normalize(g_srcHist, None, 0, 256, cv2.NORM_MINMAX)
# g_dstHist_norm = cv2.normalize(g_dstHist, None, 0, 256, cv2.NORM_MINMAX)
#
# r_srcHist_norm = cv2.normalize(r_srcHist, None, 0, 256, cv2.NORM_MINMAX)
# r_dstHist_norm = cv2.normalize(r_dstHist, None, 0, 256, cv2.NORM_MINMAX)

# # 배경 제거
# srcHist[0] = [0]
# dstHist[0] = [0]

# 누적 분포 함수 계산
# srcCdf = srcHist.copy()
# dstCdf = dstHist.copy()
# for i in range(1, histSize):
#     srcCdf[i] += srcCdf[i - 1]
#     dstCdf[i] += dstCdf[i - 1]

srcCdf = np.cumsum(srcHist)
dstCdf = np.cumsum(dstHist)

# srcCdf_norm = cv2.normalize(srcCdf, None, 0, 1, cv2.NORM_MINMAX)
# dstCdf_norm = cv2.normalize(dstCdf, None, 0, 1, cv2.NORM_MINMAX)

# g_srcCdf = g_srcHist.copy()
# g_dstCdf = g_dstHist.copy()
# for i in range(1, histSize):
#     g_srcCdf[i] += g_srcCdf[i - 1]
#     g_dstCdf[i] += g_dstCdf[i - 1]

g_srcCdf = np.cumsum(g_srcHist)
g_dstCdf = np.cumsum(g_dstHist)

# g_srcCdf_norm = cv2.normalize(g_srcCdf, None, 0, 1, cv2.NORM_MINMAX)
# g_dstCdf_norm = cv2.normalize(g_dstCdf, None, 0, 1, cv2.NORM_MINMAX)

# r_srcCdf = r_srcHist.copy()
# r_dstCdf = r_dstHist.copy()
# for i in range(1, histSize):
#     r_srcCdf[i] += r_srcCdf[i - 1]
#     r_dstCdf[i] += r_dstCdf[i - 1]

r_srcCdf = np.cumsum(r_srcHist)
r_dstCdf = np.cumsum(r_dstHist)

# r_srcCdf_norm = cv2.normalize(r_srcCdf, None, 0, 1, cv2.NORM_MINMAX)
# r_dstCdf_norm = cv2.normalize(r_dstCdf, None, 0, 1, cv2.NORM_MINMAX)


# srcCdf = (255 * srcCdf / srcCdf[-1]).astype(np.uint8)
# dstCdf = (255 * dstCdf / dstCdf[-1]).astype(np.uint8)
# g_srcCdf = (255 * g_srcCdf / g_srcCdf[-1]).astype(np.uint8)
# g_dstCdf = (255 * g_dstCdf / g_dstCdf[-1]).astype(np.uint8)
# r_srcCdf = (255 * r_srcCdf / r_srcCdf[-1]).astype(np.uint8)
# r_dstCdf = (255 * r_dstCdf / r_dstCdf[-1]).astype(np.uint8)

print(srcCdf[-1])
print(dstCdf[-1])

# 히스토그램 매칭
lut = np.zeros(256, dtype=np.uint8)
j = 0
for i in range(256):
    while j < 256 and srcCdf[i] > dstCdf[j]:
        j += 1
    lut[i] = j

g_lut = np.zeros(256, dtype=np.uint8)
j = 0
for i in range(256):
    while j < 256 and g_srcCdf[i] > g_dstCdf[j]:
        j += 1
    g_lut[i] = j

r_lut = np.zeros(256, dtype=np.uint8)
j = 0
for i in range(256):
    while j < 256 and r_srcCdf[i] > r_dstCdf[j]:
        j += 1
    r_lut[i] = j

# lut = np.zeros((1, 256), dtype=np.uint8)
# for i in range(histSize):
#     minDiff = float('inf')
#     index = 0
#     for j in range(histSize):
#         # diff = abs(srcCdf_norm[i] - dstCdf_norm[j])
#         diff = abs(srcCdf[i] - dstCdf[j])
#         # print(i, j, diff)
#         if diff < minDiff:
#             minDiff = diff
#             index = j
#             lut[0, i] = index
    # print(i, lut[0, i])

# sorted_dstHist = sorted(dstHist, reverse=True)
# sorted_srcHist = sorted(srcHist, reverse=True)
# for i in range(histSize):
#     src_index = np.where(srcHist == sorted_srcHist[i])[0]
#     dst_index = np.where(dstHist == sorted_dstHist[i])[0]
#     lut[0, src_index[0]] = dst_index[0]

# g_lut = np.zeros((1, 256), dtype=np.uint8)
# for i in range(histSize):
#     minDiff = float('inf')
#     index = 0
#     for j in range(histSize):
#         # diff = abs(g_srcCdf_norm[i] - g_dstCdf_norm[j])
#         diff = abs(g_srcCdf[i] - g_dstCdf[j])
#         if diff < minDiff:
#             minDiff = diff
#             index = j
#             g_lut[0, i] = index

# sorted_g_dstHist = sorted(g_dstHist, reverse=True)
# sorted_g_srcHist = sorted(g_srcHist, reverse=True)
# for i in range(histSize):
#     g_src_index = np.where(g_srcHist == sorted_g_srcHist[i])[0]
#     g_dst_index = np.where(g_dstHist == sorted_g_dstHist[i])[0]
#     g_lut[0, g_src_index[0]] = g_dst_index[0]

# r_lut = np.zeros((256, 1), dtype=np.uint8)
# for i in range(histSize):
#     minDiff = float('inf')
#     index = 0
#     for j in range(histSize):
#         # diff = abs(r_srcCdf_norm[i] - r_dstCdf_norm[j])
#         diff = abs(r_srcCdf[i] - r_dstCdf[j])
#         if diff < minDiff:
#             minDiff = diff
#             index = j
#             r_lut[0, i] = index

# for i in range(histSize):
#     diff = abs(r_srcCdf[i] - r_dstCdf)
#     j = np.argmin(diff)
#     r_lut[i] = j

# sorted_r_dstHist = sorted(r_dstHist, reverse=True)
# sorted_r_srcHist = sorted(r_srcHist, reverse=True)
# for i in range(histSize):
#     r_src_index = np.where(r_srcHist == sorted_r_srcHist[i])[0]
#     r_dst_index = np.where(r_dstHist == sorted_r_dstHist[i])[0]
#     r_lut[0, r_src_index[0]] = r_dst_index[0]
# print(lut)

print(len(original_b[not_zero_index]))

# 결과 이미지 생성
# resultYCrCb = cv2.LUT(original_b[not_zero_index], lut)
resultYCrCb = cv2.LUT(srcImage[:,:,0], lut)
# 결과 이미지 생성
g_result = cv2.LUT(srcImage[:,:,1], g_lut)
# 결과 이미지 생성
r_result = cv2.LUT(srcImage[:,:,2], r_lut)

result_b_ravel = resultYCrCb.ravel()
result_g_ravel = g_result.ravel()
result_r_ravel = r_result.ravel()
print(resultYCrCb.shape)

resultChannels = [resultYCrCb, g_result, r_result]
resultImage = cv2.merge(resultChannels)

# cv2.imwrite("out.png", resultImage)

cv2.imshow('original', srcImage)
cv2.imshow('target', dstImage)
cv2.imshow('out', resultImage)

plt.figure("input")
plt.subplot(1, 3, 1)
# plt.hist(srcHist, 256, [1, 256])
plt.plot(srcHist)
plt.subplot(1, 3, 2)
# plt.hist(g_srcHist, 256, [1, 256])
plt.plot(g_srcHist)
plt.subplot(1, 3, 3)
# plt.hist(r_srcHist, 256, [1, 256])
plt.plot(r_srcHist)

plt.figure("target")
plt.subplot(1, 3, 1)
# plt.hist(dstHist, 256, [1, 256])
plt.plot(dstHist)
plt.subplot(1, 3, 2)
# plt.hist(g_dstHist, 256, [1, 256])
plt.plot(g_dstHist)
plt.subplot(1, 3, 3)
# plt.hist(r_dstHist, 256, [1, 256])
plt.plot(r_dstHist)
# plt.show()

plt.figure("result")
plt.subplot(1, 3, 1)
# plt.hist(result_b_ravel, 256, [1, 256])
plt.plot(resultYCrCb)
plt.subplot(1, 3, 2)
# plt.hist(result_g_ravel, 256, [1, 256])
plt.plot(g_result)
plt.subplot(1, 3, 3)
# plt.hist(result_r_ravel, 256, [1, 256])
plt.plot(r_result)
plt.show()


cv2.waitKey()