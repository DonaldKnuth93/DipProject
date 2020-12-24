import cv2
from matplotlib import pyplot as plt

eggImg = cv2.imread("img/egg2.jpg")

eggImgColor = cv2.cvtColor(eggImg, cv2.COLOR_BGR2RGB)
grayEgg = cv2.cvtColor(eggImg, cv2.COLOR_BGR2GRAY)

# ret, threshEgg = cv2.threshold(grayEgg, 128, 255, cv2.THRESH_BINARY)
# ret, threshEggInv = cv2.threshold(grayEgg, 86, 255, cv2.THRESH_BINARY_INV)
ret, threshEggOTSU = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# ret, threshEggTriangle = cv2.threshold(grayEgg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

print(ret)

plt.figure("Original Egg Image")
plt.imshow(eggImgColor)
plt.figure("Grayscale Egg Image")
plt.imshow(grayEgg, cmap="gray")
# plt.figure("Binary Egg Image")
# plt.imshow(threshEgg, cmap="gray")
# plt.figure("Inverted Binary Egg Image")
# plt.imshow(threshEggInv, cmap="gray")
plt.figure("OTSU Inverted Egg Image")
plt.imshow(threshEggOTSU, cmap="gray")
# plt.figure("Triangle Egg Image")
# plt.imshow(threshEggTriangle, cmap="gray")
plt.show()
