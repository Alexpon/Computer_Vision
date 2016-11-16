import numpy as np
import cv2 as cv2


def binarize(img, threshold):
	lena_binarize = img.copy()
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
	        lena_binarize[i][j] = 0 if lena_binarize[i][j] < threshold else 255
	return lena_binarize

def down_sampling(img, k_size):
	o_size = img.shape[0]
	n_size = o_size/k_size
	down_img = np.zeros((n_size, n_size), dtype=int)
	for i in range(n_size):
		for j in range(n_size):
			down_img[i][j] = 1 if img[i*k_size][j*k_size]==255 else 0
	return down_img

def yokoi(img):
	size = img.shape[0]
	ans = np.zeros((size, size), dtype=int)
	for i in range(size):
		for j in range(size):
			if img[i][j]==1:
				a1 = h_func(img, [i,j], [i,j+1], [i-1,j+1], [i-1,j])
				a2 = h_func(img, [i,j], [i-1,j], [i-1,j-1], [i,j-1])
				a3 = h_func(img, [i,j], [i,j-1], [i+1,j-1], [i+1,j])
				a4 = h_func(img, [i,j], [i+1,j], [i+1,j+1], [i,j+1])
				a  = a1+a2+a3+a4
				ans[i][j]=5 if a==40 else a%10
	return ans

def h_func(img, x1, x2, x3, x4):
	size = img.shape[0]
	if x2[0]>=size or x2[1]>=size or x2[0]<0 or x2[1]<0:
		return 0
	elif (img[x1[0]][x1[1]]) != (img[x2[0]][x2[1]]):
		return 0
	if x3[0]>=size or x3[1]>=size or x3[0]<0 or x3[1]<0:
		return 1
	elif img[x3[0]][x3[1]]==img[x1[0]][x1[1]] and img[x4[0]][x4[1]]==img[x1[0]][x1[1]]:
		return 10
	else:
		return 1

def write2txt(img):
	f = open('yokoi_num.txt', 'w')
	size = img.shape[0]
	for i in range(size):
		for j in range(size):
			if img[i][j]==0:
				f.write("  ")
			else:
				f.write(str(img[i][j])+" ")
		f.write('\n')

def main():
	img = cv2.imread('lena.bmp', 0)
	threshold = 128
	lena_binarize = binarize(img, threshold)
	lena_down = down_sampling(lena_binarize, 8)
	ans = yokoi(lena_down)
	write2txt(ans)

if __name__ == '__main__':
    main()

