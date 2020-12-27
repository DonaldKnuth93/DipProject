# 1- input RGB Image
# 2- convert to graysclae
# 3- localizing the egg
# 4- image enhancement and filtering :
#   +localize the image and output the subimage
#   +increase the level of visibility inside the embryo we increase the omage contrast by applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to equalize image
#   +
# 5- dynamic thresholding
# 6- building binary image
# 7- classification


import cv2
import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from PIL import Image, ImageFilter

# Reading the image from the present directory
ordinary_image = cv2.imread("img/egg1.jpg")
resized_ordinary_image = cv2.resize(ordinary_image, (500, 600))

# Resizing the image for compatibility
image = cv2.resize(ordinary_image, (500, 600))

# The initial processing of the image
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image_bw, 3)
kernel = np.ones((5, 5), np.float32) / 25
dst = cv.filter2D(image_bw, -1, kernel)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
eggClahe = cv2.createCLAHE(clipLimit=5)
final_img = eggClahe.apply(image_bw) + 30

# Ordinary thresholding the same image
ret, threshEggOTSU = cv2.threshold(image_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)

# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
edges = cv2.Canny(threshEggOTSU, 100, 200)


plt.subplot(2, 3, 1), plt.imshow(resized_ordinary_image, cmap='gray')
plt.title('ordinary image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(image, cmap='gray')
plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(final_img, cmap='gray')
plt.title('CLAHE image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(threshEggOTSU, cmap='gray')
plt.title('OTSU Threshold image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(edges, cmap='gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(ordinary_image, cmap='gray')
plt.title('Hessian Matrix'), plt.xticks([]), plt.yticks([])
plt.show()

# cv2.imshow('', ordinary_image)
# cv2.waitKey(0)
