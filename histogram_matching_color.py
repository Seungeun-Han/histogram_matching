import cv2
import matplotlib.pyplot as plt
import numpy as np


# load image
img = cv2.imread('C:/Users/hsyal/Desktop/etrimask_256_results_0/000004-005.png', cv2.IMREAD_COLOR)
target = cv2.imread('C:/Users/hsyal/Desktop/etri_maskDB/mask_skincolor_images/000004-005.jpg', cv2.IMREAD_COLOR)

# split
img_b, img_g, img_r = cv2.split(img)
target_b, target_g, target_r = cv2.split(target)

shape = img_b.shape
original_b = img_b.ravel()
specified_b = target_b.ravel()
original_g = img_g.ravel()
specified_g = target_g.ravel()
original_r = img_r.ravel()
specified_r = target_r.ravel()


b_values, b_idx, b_counts = np.unique(original_b, return_inverse=True, return_counts=True)
bt_values, bt_counts = np.unique(specified_b, return_counts=True)
g_values, g_idx, g_counts = np.unique(original_g, return_inverse=True, return_counts=True)
gt_values, gt_counts = np.unique(specified_g, return_counts=True)
r_values, r_idx, r_counts = np.unique(original_r, return_inverse=True, return_counts=True)
rt_values, rt_counts = np.unique(specified_r, return_counts=True)


b_quantiles = np.cumsum(b_counts).astype(np.float64)
b_quantiles /= b_quantiles[-1]
b_sour = np.around(b_quantiles * 255)
bt_quantiles = np.cumsum(bt_counts).astype(np.float64)
bt_quantiles /= bt_quantiles[-1]
b_temp = np.around(bt_quantiles * 255)

g_quantiles = np.cumsum(g_counts).astype(np.float64)
g_quantiles /= g_quantiles[-1]
g_sour = np.around(g_quantiles * 255)
gt_quantiles = np.cumsum(gt_counts).astype(np.float64)
gt_quantiles /= gt_quantiles[-1]
g_temp = np.around(gt_quantiles * 255)

r_quantiles = np.cumsum(r_counts).astype(np.float64)
r_quantiles /= r_quantiles[-1]
r_sour = np.around(r_quantiles * 255)
rt_quantiles = np.cumsum(rt_counts).astype(np.float64)
rt_quantiles /= rt_quantiles[-1]
r_temp = np.around(rt_quantiles * 255)


b_list = []
for data in b_sour:
    diff = b_temp - data
    mask = np.ma.less_equal(diff, -1)
    if np.all(mask):
        c = np.abs(diff).argmin()
        b_list.append(c)
    masked_diff = np.ma.masked_array(diff, mask)
    b_list.append(masked_diff.argmin())
LUT = np.array(b_list, dtype='uint8')
b_out = np.array(LUT[b_idx].reshape(shape))

g_list = []
for data in g_sour:
    diff = g_temp - data
    mask = np.ma.less_equal(diff, -1)
    if np.all(mask):
        c = np.abs(diff).argmin()
        g_list.append(c)
    masked_diff = np.ma.masked_array(diff, mask)
    g_list.append(masked_diff.argmin())
LUT = np.array(g_list, dtype='uint8')
g_out = np.array(LUT[g_idx].reshape(shape))

r_list = []
for data in r_sour:
    diff = r_temp - data
    mask = np.ma.less_equal(diff, -1)
    if np.all(mask):
        c = np.abs(diff).argmin()
        r_list.append(c)
    masked_diff = np.ma.masked_array(diff, mask)
    r_list.append(masked_diff.argmin())
LUT = np.array(r_list, dtype='uint8')
r_out = np.array(LUT[r_idx].reshape(shape))

out = cv2.merge((b_out, g_out, r_out))  # b_out, g_out, r_out, img_b, img_g, img_r

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

cv2.waitKey()