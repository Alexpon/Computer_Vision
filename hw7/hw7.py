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

def interior_border(down_img):
	# 0: nothing
	# 1: interior (i)
	# 2: border   (b)
	size = down_img.shape[0]
	int_bor = np.zeros((size, size), dtype=int)
	for i in range(down_img.shape[0]):
		for j in range(down_img.shape[1]):
			if(down_img[i][j]!=0):
				int_bor[i][j] = 1
				for m in [-1, 0, 1]:
					for n in [-1, 0, 1]:
						if (i+m<0 or i+m>=down_img.shape[0] or j+n<0 or j+n>=down_img.shape[1]):
							int_bor[i][j] = 2
							break
						elif down_img[i+m][j+n]!=1:
							int_bor[i][j] = 2
			else:
				int_bor[i][j] = 0
	return int_bor

def pair_relation(int_bor):
	# 0: nothing
	# 1: interior      (i)
	# 2: border        (b)
	# 3: marked border (m)
	for i in range(int_bor.shape[0]):
		for j in range(int_bor.shape[1]):
			if int_bor[i][j]==1:
				for m in [-1, 0, 1]:
					for n in [-1, 0, 1]:
						if(int_bor[i+m][j+n]!=1):
							int_bor[i+m][j+n]=3
	return int_bor

def shrink(mark_int_bor):
	for i in range(mark_int_bor.shape[0]):
		for j in range(mark_int_bor.shape[1]):
			if mark_int_bor[i][j]==3:
				mark_int_bor[i][j]=yokoi(mark_int_bor, i, j)
	return mark_int_bor


def yokoi(img, i, j):
	a1 = h_func(img, [i,j], [i,j+1], [i-1,j+1], [i-1,j])
	a2 = h_func(img, [i,j], [i-1,j], [i-1,j-1], [i,j-1])
	a3 = h_func(img, [i,j], [i,j-1], [i+1,j-1], [i+1,j])
	a4 = h_func(img, [i,j], [i+1,j], [i+1,j+1], [i,j+1])
	a  = a1+a2+a3+a4
	return 1 if a>1 else 0

def h_func(img, x1, x2, x3, x4):
	size = img.shape[0]
	if x2[0]>=size or x2[1]>=size or x2[0]<0 or x2[1]<0:
		return 0
	if (img[x2[0]][x2[1]]==0):
		return 0
	if x3[0]>=size or x3[1]>=size or x3[0]<0 or x3[1]<0:
		return 1
	if x4[0]>=size or x4[1]>=size or x4[0]<0 or x4[1]<0:
		return 1
	elif img[x3[0]][x3[1]]==0 or img[x4[0]][x4[1]]==0:
		return 1
	else:
		return 0

def pixel_count(img):
	cnt = 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]!=0):
				cnt += 1
	return cnt

def write2txt(img):
	f = open('lena_thinning.txt', 'w')
	size = img.shape[0]
	for i in range(size):
		for j in range(size):
			if img[i][j]!=0:
				f.write("1 ")
			else:
				f.write("  ")
		f.write('\n')

def write2bmp(img):
	size = img.shape[0]
	for i in range(size):
		for j in range(size):
			if img[i][j]!=0:
				img[i][j]=255
	cv2.imwrite('lena_thinning.bmp',img)

def main():
	img = cv2.imread('lena.bmp', 0)
	threshold = 128
	lena_binarize = binarize(img, threshold)
	lena = down_sampling(lena_binarize, 8)
	
	while True:
		pre_count = pixel_count(lena)
		int_bor = interior_border(lena)
		mark_int_bor = pair_relation(int_bor)
		lena = shrink(mark_int_bor)
		post_count = pixel_count(lena)
		if pre_count==post_count:
			break

	write2txt(lena)
	write2bmp(lena)


if __name__ == '__main__':
    main()

