import numpy as np
import cv2 as cv2


def binarize(img, threshold):
	lena_binarize = img.copy()
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
	        lena_binarize[i][j] = 0 if lena_binarize[i][j] < threshold else 255
	return lena_binarize

def dilation(img, kernel):
	lena_dilation = np.zeros(img.shape, dtype=np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]==255):
				for k, l in kernel:
					if ((i+k)>=0 and (j+l)>=0 and (i+k)<img.shape[0] and (j+l)<img.shape[1]):
						lena_dilation[i+k][j+l] = 255
	return lena_dilation

def erosion(img, kernel):
	lena_erosion = np.zeros(img.shape, dtype=np.int)
	center = 255 if [0,0] in kernel else 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]==center):
				fit = True
				for k, l in kernel:
					if ((i+k)<0 or (j+l)<0 or (i+k)>=img.shape[0] or (j+l)>=img.shape[1] or img[i+k][j+l]==0):
						fit = False
						break
				if fit:
					lena_erosion[i][j] = 255;
	return lena_erosion

def opening(img, kernel):
	return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
	return erosion(dilation(img, kernel), kernel)

def hit_miss(img, J, K):
	com_img = np.zeros(img.shape, dtype=np.int)
	hit_miss_img = np.zeros(img.shape, dtype=np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]==0):
				com_img[i][j]=255
	er1 = erosion(img, J)
	er2 = erosion(com_img, K)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(er1[i][j] and er2[i][j] == 255):
				hit_miss_img[i][j] = 255
	return hit_miss_img

def main():
	kernel = [[-1,0],[0,1],[0,0],[0,-1],[1,0]]
	J = [[0,-1],[0,0],[1,0]]
	K = [[-1,0],[-1,1],[0,1]]

	img = cv2.imread('lena.bmp', 0)
	threshold = 128
	lena_binarize = binarize(img, threshold)

	lena_dilation = dilation(lena_binarize, kernel)
	lena_erosion = erosion(lena_binarize, kernel)
	lena_opening = opening(lena_binarize, kernel)
	lena_closing = closing(lena_binarize, kernel)
	lena_hit_miss = hit_miss(lena_binarize, J, K)
	
	cv2.imwrite('lena_dilation.bmp',lena_dilation)
	cv2.imwrite('lena_erosion.bmp',lena_erosion)
	cv2.imwrite('lena_opening.bmp',lena_opening)
	cv2.imwrite('lena_closing.bmp',lena_closing)
	cv2.imwrite('lena_hit_miss.bmp',lena_hit_miss)


if __name__ == '__main__':
    main()

