import cv2
from matplotlib import pyplot as plt
from numpy.core.defchararray import lower, upper

from processing import sobel_filter
from tumor_extraction import median_filtered

eggImg = cv2.imread("img/egg2.jpg")

eggImgColor = cv2.cvtColor(eggImg, cv2.COLOR_BGR2RGB)
grayEgg = cv2.cvtColor(eggImg, cv2.COLOR_BGR2GRAY)

# ret, threshEgg = cv2.threshold(grayEgg, 128, 255, cv2.THRESH_BINARY)
# ret, threshEggInv = cv2.threshold(grayEgg, 86, 255, cv2.THRESH_BINARY_INV)
ret, threshEggOTSU = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, threshEggTriangle = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
print(ret)

# Step 3a - apply Sobel filter
img_sobelx = sobel_filter(median_filtered, 1, 0)
img_sobely = sobel_filter(median_filtered, 0, 1)
# Adding mask to the image
img_sobel = img_sobelx + img_sobely + grayEgg


plt.figure("Original Egg Image")
plt.imshow(eggImgColor)
plt.figure("Grayscale Egg Image")
plt.imshow(grayEgg, cmap="gray")
# plt.figure("Binary Egg Image")
# plt.imshow(threshEgg, cmap="gray")
# plt.figure("Inverted Binary Egg Image")
# plt.imshow(threshEggInv, cmap="gray")
plt.figure("OTSU Egg Image")
plt.imshow(threshEggOTSU, cmap="gray")
# plt.figure("Triangle Egg Image")
# plt.imshow(threshEggTriangle, cmap="gray")
plt.figure("Sobel Egg Image")
plt.imshow(img_sobel, cmap="gray")

plt.show()
