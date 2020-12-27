import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

eggImg = cv2.imread("img/egg1.jpg")

eggImgColor = cv2.cvtColor(eggImg, cv2.COLOR_BGR2RGB)
grayEgg = cv2.cvtColor(eggImg, cv2.COLOR_BGR2GRAY)


# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(grayEgg,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)


# eggLaplacian = cv.Laplacian(sobel_8u, cv.CV_64F)
eggSobelX = cv.Sobel(sobel_8u, cv.CV_64F, 1, 0, ksize=3)
eggSobelY = cv.Sobel(sobel_8u, cv.CV_64F, 0, 1, ksize=3)

# # Adding mask to the image
eggSobelMask = eggSobelX + eggSobelY + sobel_8u


# apply_threshold function implementation
def apply_threshold(image, **kwargs):
    '''
    :param image: image object
    :param kwargs: threshold parameters - dictionary
    :return:
    '''
    threshold_method = kwargs['threshold_method']
    max_value = kwargs['pixel_value']
    threshold_flag = kwargs.get('threshold_flag', None)
    if threshold_flag is not None:
        ret, thresh1 = cv2.adaptiveThreshold(image, max_value, threshold_method, cv2.THRESH_BINARY,
                                             kwargs['block_size'], kwargs['const'])
    else:
        ret, thresh1 = cv2.threshold(image, kwargs['threshold'], max_value, threshold_method)
    return thresh1


# Threshold the pixel values
eggThresh = apply_threshold(grayEgg, **{"threshold": 160,
                                             "pixel_value": 255,
                                             "threshold_method": cv2.THRESH_BINARY})

ret, eggThreshInv = cv2.threshold(eggThresh, 0, 255, cv2.THRESH_BINARY_INV)
print(ret)

plt.subplot(3, 3, 1), plt.imshow(grayEgg, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 2), plt.imshow(sobel_8u, cmap='gray')
plt.title('sobel_8u'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(eggThresh, cmap='gray')
plt.title('Threshold'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(eggThreshInv, cmap='gray')
plt.title('Inverted Threshold'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 3, 2), plt.imshow(eggLaplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 3, 3), plt.imshow(eggSobelX, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 3, 4), plt.imshow(eggSobelX, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(eggSobelMask, cmap='gray')
plt.title('Sobel Mask'), plt.xticks([]), plt.yticks([])

plt.show()

# ret, threshEgg = cv2.threshold(grayEgg, 128, 255, cv2.THRESH_BINARY)
# ret, threshEggInv = cv2.threshold(grayEgg, 86, 255, cv2.THRESH_BINARY_INV)
# ret, threshEggOTSU = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, threshEggTriangle = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
# print(ret)


# plt.figure("Original Egg Image")
# plt.imshow(eggImgColor)
# plt.figure("Grayscale Egg Image")
# plt.imshow(grayEgg, cmap="gray")
# plt.figure("Binary Egg Image")
# plt.imshow(threshEgg, cmap="gray")
# plt.figure("Inverted Binary Egg Image")
# plt.imshow(threshEggInv, cmap="gray")
# plt.figure("OTSU Egg Image")
# plt.imshow(threshEggOTSU, cmap="gray")
# plt.figure("Triangle Egg Image")
# plt.imshow(threshEggTriangle, cmap="gray")
