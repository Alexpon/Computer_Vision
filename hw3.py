import numpy as np
import cv2 as cv2

def counting(img):
	lena_binarize = img.copy()
	counter = np.zeros(256)
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
	        rgb = lena_binarize[i][j][0]
	        counter[rgb] += 1
	return counter

def cumulative(counter):
    for i in range(1,len(counter)):
        counter[i] += counter[i-1]
    return counter/counter[len(counter)-1]

def his_equalization(img, cumu_cnt):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rgb = img[i][j][0]
            equ_val = 255*cumu_cnt[rgb]
            img[i][j] = [equ_val, equ_val, equ_val]
    return img

img = cv2.imread('lena.bmp')
counter = counting(img)
cumu_cnt = cumulative(counter)
lena_equ = his_equalization(img, cumu_cnt)

cv2.imwrite('lena_equ.bmp',lena_equ)
cv2.imshow('image', lena_equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

