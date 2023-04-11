import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image
img = cv2.imread('C:/Users/hsyal/Desktop/etrimask_256_results_0/000004-005.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('C:/Users/hsyal/Desktop/etri_maskDB/mask_skincolor_images/000004-005.jpg', cv2.IMREAD_GRAYSCALE)



shape = img.shape
original = img.ravel()
specified = target.ravel()


s_values, bin_idx, s_counts = np.unique(original, return_inverse=True, return_counts=True)
t_values, t_counts = np.unique(specified, return_counts=True)


s_quantiles = np.cumsum(s_counts).astype(np.float64)
s_quantiles /= s_quantiles[-1]
sour = np.around(s_quantiles * 255)
t_quantiles = np.cumsum(t_counts).astype(np.float64)
t_quantiles /= t_quantiles[-1]
temp = np.around(t_quantiles * 255)


b = []
for data in sour:
    diff = temp - data
    mask = np.ma.less_equal(diff, -1)
    if np.all(mask):
        c = np.abs(diff).argmin()
        b.append(c)
    masked_diff = np.ma.masked_array(diff, mask)
    b.append(masked_diff.argmin())
LUT = np.array(b, dtype='uint8')
out = np.array(LUT[bin_idx].reshape(shape))


cv2.imshow('original', img)
cv2.imshow('target', target)
cv2.imshow('out', out)

plt.figure()
plt.subplot(1, 3, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.subplot(1, 3, 2)
plt.hist(target.ravel(), 256, [0, 256])
plt.subplot(1, 3, 3)
plt.hist(out.ravel(), 256, [0, 256])
plt.show()

cv2.waitKey(0)