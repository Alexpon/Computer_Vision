import numpy as np
import cv2 as cv2


def binarize(img, threshold):
	lena_binarize = img.copy()
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
	        rgb = lena_binarize[i][j][0]
	        if rgb < threshold:
	            lena_binarize[i][j] = [0,0,0]
	        else:
	            lena_binarize[i][j] = [255,255,255]
	return lena_binarize

def dilation(img, kernel):
	lena_dilation = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j][0]==255):
				for k, l in kernel:
					if ((i+k)>=0 and (j+l)>=0 and (i+k)<img.shape[0] and (j+l)<img.shape[1]):
						lena_dilation[i+k][j+l] = [255,255,255]
	return lena_dilation

def erosion(img, kernel):
	lena_erosion = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			lena_erosion[i][j] = [0,0,0]

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j][0]==255):
				fit = True
				for k, l in kernel:
					if ((i+k)<0 or (j+l)<0 or (i+k)>=img.shape[0] or (j+l)>=img.shape[1] or img[i+k][j+l][0]==0):
						fit = False
						break
				if fit:
					lena_erosion[i+k][j+l] = [255,255,255];
	return lena_erosion

def opening(img, kernel):
	return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
	return erosion(dilation(img, kernel), kernel)

#def hit_mis(img, J, K):


kernel = [[-1,0],[0,1],[0,0],[0,-1],[1,0]]
J = [[0,-1],[0,0],[1,0]]
K = [[-1,0],[-1,1],[0,1]]

img = cv2.imread('lena.bmp')
threshold = 128
lena_binarize = binarize(img, threshold)

lena_dilation = dilation(lena_binarize, kernel)
lena_erosion = erosion(lena_binarize, kernel)
lena_opening = opening(lena_binarize, kernel)
lena_closing = closing(lena_binarize, kernel)
#lena_hit_miss = img.copy()

#cv2.imwrite('lena_binarize.bmp',lena_binarize)
cv2.imshow('image', lena_opening)
cv2.waitKey(0)
cv2.destroyAllWindows()


