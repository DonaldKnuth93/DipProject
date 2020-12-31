# 1- input RGB Image
# 2- convert to graysclae
# 3- localizing the egg
# 4- image enhancement and filtering :
#   +localize the image and output the subimage
#   +increase the level of visibility inside the embryo we increase the omage contrast by applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to equalize image
# 5- dynamic thresholding
# 6- building binary image
# 7- classification


import cv2
import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

# Reading the image from the present directory
ordinary_image = cv2.imread("img/egg1.jpg")
resized_ordinary_image = cv2.resize(ordinary_image, (500, 600))

# Resizing the image for compatibility
image = cv2.resize(ordinary_image, (500, 600))

# The initial processing of the image
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image_bw, 3)
gussicanEgg = cv2.GaussianBlur(image, (3, 3), 0)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
eggClahe = cv2.createCLAHE(clipLimit=5)
final_img = eggClahe.apply(image_bw) + 30

# convolute with proper kernels
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  #
# Ordinary thresholding the same image
ret, threshEggOTSU = cv2.threshold(image_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)

# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
edges = cv2.Canny(threshEggOTSU, 100, 200)


plt.subplot(2, 3, 1), plt.imshow(resized_ordinary_image, cmap='gray')
plt.title('ordinary image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(gussicanEgg, cmap='gray')
plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(final_img, cmap='gray')
plt.title('CLAHE image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(threshEggOTSU, cmap='gray')
plt.title('OTSU Threshold image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(edges, cmap='gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
47
# cv2.imshow('', ordinary_image)
# cv2.waitKey(0)
