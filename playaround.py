import cv2 as cv
import numpy as np
import math as m
import imutils

# Cancel the red ones
# yellow is the least damage ones
# orange is the more damage ones
 
image = cv.imread(r'C:/Users/aravs/OneDrive/Desktop/sudokuc1.png')
cv.imshow("image",image)

grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow("grey",grey_image)

blur = cv.GaussianBlur(grey_image, (5,5), 0)
cv.imshow("blur", blur)

threshold_image = cv.adaptiveThreshold(grey_image, 255, 1,1,11,2)
cv.imshow("Threshold",threshold_image)

contour, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_area = 0
c = 0 

for i in contour:
    area = cv.contourArea(i)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = i
            image1 = cv.drawContours(image, contour, c, (255,0,0), 3)
    c += 1

cv.imshow("Contour", image1)
print(contour)

mask = np.zeros((grey_image.shape),np.uint8)
cv.drawContours(mask,[best_cnt],0,255,-1)
cv.drawContours(mask,[best_cnt],0,0,2)
cv.imshow("mask",mask)

out = np.zeros_like(grey_image)
out[mask == 255] = grey_image[mask == 255]
cv.imshow("Iteration",out)

cnts = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL,
cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv.drawContours(image, [c], -1, (0, 255, 255), 2)
cv.circle(image, extLeft, 8, (0, 0, 255), -1)
cv.circle(image, extRight, 8, (0, 255, 0), -1)
cv.circle(image, extTop, 8, (255, 0, 0), -1)
cv.circle(image, extBot, 8, (255, 255, 0), -1)
# show the output image
cv.imshow("Image", image)

blur = cv.GaussianBlur(out, (5,5), 0)
cv.imshow("blur1", blur)
threshold_image = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
cv.imshow("thresh1", threshold_image)

contours, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

c = 0
for i in contours:
        area = cv.contourArea(i)
        if area > 1000/2:
            cv.drawContours(image, contours, c, (0, 255, 0), 3)
        c+=1

cv.imshow("Final Image", image)
print(extTop, extBot, extLeft, extRight)
cropped_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
cv.imshow("cropped_img", cropped_img[10:55,10:55])
cv.imshow("cropped_image",cropped_img)

grid_HSV = cv.cvtColor(cropped_img,cv.COLOR_RGB2HSV)
lower = np.array([0,0,0])
upper = np.array([5,5,5])
mask1 = cv.inRange(grid_HSV,lower,upper)
cv.imshow("detect",mask1)

cv.waitKey(0)
cv.destroyAllWindows()

cv.waitKey(0)