from processing import *

image = cv2.imread("./img/egg3.jpg", 1)
plt.imshow(image)

# Step one - grayscale the image
grayscale_img = cvt_image_colorspace(image)
plt.imshow(grayscale_img)

# Step two - filter out image
median_filtered = median_filtering(grayscale_img, 5)
plt.imshow(median_filtered)

# testing threshold function
bin_image = apply_threshold(median_filtered, **{"threshold": 160,
                                                "pixel_value": 255,
                                                "threshold_method": cv2.THRESH_BINARY})
otsu_image = apply_threshold(median_filtered, **{"threshold": 0,
                                                 "pixel_value": 255,
                                                 "threshold_method": cv2.THRESH_BINARY + cv2.THRESH_OTSU})

'''
th2 = cv2.adaptiveThreshold(median_filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(median_filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
show_image('Image after adaptive thresholding', th2)
show_image('Image after adaptive thresholding 2', th3)
'''

# Step 3a - apply Sobel filter
img_sobelx = sobel_filter(median_filtered, 1, 0)
img_sobely = sobel_filter(median_filtered, 0, 1)

# Adding mask to the image
img_sobel = img_sobelx + img_sobely + grayscale_img
plt.imshow(img_sobel)

# Step 4 - apply threshold
# Set threshold and maxValue
threshold = 160
maxValue = 255

# Threshold the pixel values
thresh = apply_threshold(img_sobel, **{"threshold": 160,
                                       "pixel_value": 255,
                                       "threshold_method": cv2.THRESH_BINARY})
plt.imshow(thresh)

# Step 3b - apply erosion + dilation
# apply erosion and dilation to show only the part of the image having more intensity - tumor region
# that we want to extract
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
erosion = cv2.morphologyEx(median_filtered, cv2.MORPH_ERODE, kernel)
plt.imshow(erosion)

dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel)
plt.imshow(dilation)

# Step 4 - apply thresholding
threshold = 160
maxValue = 255

# apply thresholding
new_thresholding = apply_threshold(dilation, **{"threshold": 160,
                                                "pixel_value": 255,
                                                "threshold_method": cv2.THRESH_BINARY})
plt.imshow(new_thresholding)
plt.show()
