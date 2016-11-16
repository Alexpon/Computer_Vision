import numpy as np
import cv2 as cv2

def dilation(img, kernel):
	lena_dilation = np.zeros(img.shape, dtype=np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]>0):
				max_pixel = 0
				for k, l in kernel:
					if ((i+k)>=0 and (j+l)>=0 and (i+k)<img.shape[0] and (j+l)<img.shape[1]):
						max_pixel = img[i+k][j+l] if img[i+k][j+l] > max_pixel else max_pixel

				for k, l in kernel:
					if ((i+k)>=0 and (j+l)>=0 and (i+k)<img.shape[0] and (j+l)<img.shape[1]):
						lena_dilation[i+k][j+l] = max_pixel
	return lena_dilation

def erosion(img, kernel):
	lena_erosion = np.zeros(img.shape, dtype=np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]>0):
				fit = True
				min_pixel = 255
				for k, l in kernel:
					if ((i+k)<0 or (j+l)<0 or (i+k)>=img.shape[0] or (j+l)>=img.shape[1] or img[i+k][j+l]==0):
						fit = False
						break
				if fit:
					for k, l in kernel:
						min_pixel = img[i+k][j+l] if img[i+k][j+l] < min_pixel else min_pixel
					lena_erosion[i][j] = min_pixel;
	return lena_erosion

def opening(img, kernel):
	return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
	return erosion(dilation(img, kernel), kernel)


def main():
	kernel = [			  [-2, -1], [-2, 0], [-2, 1],
       			[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        		[ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
        		[ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2],
        				  [ 2, -1], [ 2, 0], [ 2, 1]
    		]

	img = cv2.imread('lena.bmp', 0)

	lena_dilation = dilation(img, kernel)
	lena_erosion = erosion(img, kernel)
	lena_opening = opening(img, kernel)
	lena_closing = closing(img, kernel)
	
	cv2.imwrite('lena_dilation.bmp',lena_dilation)
	cv2.imwrite('lena_erosion.bmp',lena_erosion)
	cv2.imwrite('lena_opening.bmp',lena_opening)
	cv2.imwrite('lena_closing.bmp',lena_closing)
	

if __name__ == '__main__':
    main()

