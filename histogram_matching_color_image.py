import cv2
import numpy as np
import matplotlib.pyplot as plt

SRC_IMAGE_PATH = "forest.png"  # ./etri_maskDB_results/mask_inpainting/000001-004.bmp
DST_IMAGE_PATH = "sea.png"  # ./etri_maskDB_results/skin/000001-004.bmp

# 히스토그램 매칭 메소드 - GPT가 작성한 알고리즘을 기반으로 보완.
def histogram_matching_fromGPT(srcImage, srcHist, dstHist):
    # 히스토그램 정규화 - 두 영상이 픽셀 값이 0이 아닌 유효 픽셀의 개수가 다르므로 정규화 작업이 필요함.
    srcHist = cv2.normalize(srcHist, None, 0, 1, cv2.NORM_MINMAX)
    dstHist = cv2.normalize(dstHist, None, 0, 1, cv2.NORM_MINMAX)

    # 히스토그램 누적 분포 계산
    srcCdf = np.cumsum(srcHist)
    dstCdf = np.cumsum(dstHist)

    srcCdf = srcCdf * dstHist.max() / srcCdf.max()
    dstCdf = dstCdf * srcHist.max() / dstCdf.max()

    # 룩업테이블(LookUp Table) 작성
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

def main():
    srcImage = cv2.imread(SRC_IMAGE_PATH)
    dstImage = cv2.imread(DST_IMAGE_PATH)


    # 각 채널에 대해 히스토그램 계산
    histSize = 256
    range_ = [0, 256]
    histRange = range_
    uniform = True
    accumulate = False
    b_srcHist = cv2.calcHist([srcImage], [0], None, [histSize], histRange, uniform, accumulate)
    b_dstHist = cv2.calcHist([dstImage], [0], None, [histSize], histRange, uniform, accumulate)
    g_srcHist = cv2.calcHist([srcImage], [1], None, [histSize], histRange, uniform, accumulate)
    g_dstHist = cv2.calcHist([dstImage], [1], None, [histSize], histRange, uniform, accumulate)
    r_srcHist = cv2.calcHist([srcImage], [2], None, [histSize], histRange, uniform, accumulate)
    r_dstHist = cv2.calcHist([dstImage], [2], None, [histSize], histRange, uniform, accumulate)

    # etri mask db를 사용한다면 값이 0인 픽셀의 개수가 매우 많으므로 제거
    if "etri" in SRC_IMAGE_PATH:
        b_srcHist[0] = 0
        b_dstHist[0] = 0
        g_srcHist[0] = 0
        g_dstHist[0] = 0
        r_srcHist[0] = 0
        r_dstHist[0] = 0


    # 각 채널에 대해 히스토그램 매칭 수행
    b_resultImage = histogram_matching_fromGPT(srcImage[:,:,0], b_srcHist, b_dstHist)
    g_resultImage = histogram_matching_fromGPT(srcImage[:,:,1], g_srcHist, g_dstHist)
    r_resultImage = histogram_matching_fromGPT(srcImage[:,:,2], r_srcHist, r_dstHist)

    # merge
    resultChannels = [b_resultImage, g_resultImage, r_resultImage]
    resultImage = cv2.merge(resultChannels)

    # calc result hist
    b_resultHist = cv2.calcHist([b_resultImage], [0], None, [histSize], histRange, uniform, accumulate)
    g_resultHist = cv2.calcHist([g_resultImage], [0], None, [histSize], histRange, uniform, accumulate)
    r_resultHist = cv2.calcHist([r_resultImage], [0], None, [histSize], histRange, uniform, accumulate)


    # cv2.imwrite("color_test_result.png", resultImage)

    cv2.imshow('original', srcImage)
    cv2.imshow('target', dstImage)
    cv2.imshow('out', resultImage)

    # 각 채널에 대해 히스토그램 매칭이 잘 수행되었는지 확인하기 위해 히스토그램 출력
    plt.figure("b_result", figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(b_srcHist)
    plt.subplot(1, 3, 2)
    plt.plot(b_dstHist)
    plt.subplot(1, 3, 3)
    plt.plot(b_resultHist[1:])

    plt.figure("g_result", figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(g_srcHist)
    plt.subplot(1, 3, 2)
    plt.plot(g_dstHist)
    plt.subplot(1, 3, 3)
    plt.plot(g_resultHist[1:])

    plt.figure("r_result", figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(r_srcHist)
    plt.subplot(1, 3, 2)
    plt.plot(r_dstHist)
    plt.subplot(1, 3, 3)
    plt.plot(r_resultHist[1:])
    plt.show()

    cv2.waitKey()

if __name__ == "__main__":
    main()