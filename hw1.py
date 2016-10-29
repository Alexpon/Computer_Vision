import numpy as np
import cv2 as cv2

img = cv2.imread('lena.bmp')

row = img.shape[0]
col = img.shape[1]
lena_ud = img.copy()
lena_lr = img.copy()
lena_ur = img.copy()

for i in range(row):
    for j in range(col):
        lena_ud[i,j] = img[row-i-1,j]
        lena_lr[i,j] = img[i,col-j-1]
        lena_ur[i,j] = img[j,i]

cv2.imwrite('lena_updown.bmp',lena_ud)
cv2.imwrite('lena_leftright.bmp',lena_lr)
cv2.imwrite('lena_mirror.bmp',lena_ur)

cv2.imshow('image', lena_ud)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image', lena_lr)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image', lena_ur)
cv2.waitKey(0)
cv2.destroyAllWindows()
