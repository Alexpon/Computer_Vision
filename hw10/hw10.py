import numpy as np
import cv2 as cv2

def main():
	img = cv2.imread('lena.bmp', 0)

	laplacian_threshold = 15
	laplacian_high_nor = 1.0/3
	laplacian_high_kernel = [
								[-1,-1, 1], [-1, 0, 1], [-1, 1, 1],
								[ 0,-1, 1], [ 0, 0,-8], [ 0, 1, 1],
								[ 1,-1, 1], [ 1, 0, 1], [ 1, 1, 1],
							]

	laplacian_low_kernel = [
								[-1,-1, 0], [-1, 0, 1], [-1, 1, 0],
								[ 0,-1, 1], [ 0, 0,-4], [ 0, 1, 1],
								[ 1,-1, 0], [ 1, 0, 1], [ 1, 1, 0],
							]

	laplacian_min_var_threshold = 20
	laplacian_min_var_nor = 1.0/3
	laplacian_min_var_kernel =	[
								[-1,-1, 2], [-1, 0,-1], [-1, 1, 2],
								[ 0,-1,-1], [ 0, 0,-4], [ 0, 1,-1],
								[ 1,-1, 2], [ 1, 0,-1], [ 1, 1, 2],
								]

	laplacian_gaussian_threshold = 3000
	laplacian_gaussian_kernel =	[
																		  [-5,-2,  -1], [-5,-1,  -1], [-5, 0,  -2], [-5, 1,  -1], [-5, 2,  -1],
															[-4,-3,  -2], [-4,-2,  -4], [-4,-1,  -8], [-4, 0,  -9], [-4, 1,  -8], [-4, 2,  -4], [-4, 3,  -2],
											  [-3,-4,  -2], [-3,-3,  -7], [-3,-2, -15], [-3,-1, -22], [-3, 0, -23], [-3, 1, -22], [-3, 2, -15], [-3, 3,  -7], [-3, 4,  -2],
								[-2,-5,  -1], [-2,-4,  -4], [-2,-3, -15], [-2,-2, -24], [-2,-1, -14], [-2, 0,  -1], [-2, 1, -14], [-2, 2, -24], [-2, 3, -15], [-2, 4,  -4], [-2, 5,  -1],
								[-1,-5,  -1], [-1,-4,  -8], [-1,-3, -22], [-1,-2, -14], [-1,-1,  52], [-1, 0, 103], [-1, 1,  52], [-1, 2, -14], [-1, 3, -22], [-1, 4,  -8], [-1, 5,  -1],
								[ 0,-5,  -2], [ 0,-4,  -9], [ 0,-3, -23], [ 0,-2,  -1], [ 0,-1, 103], [ 0, 0, 178], [ 0, 1, 103], [ 0, 2,  -1], [ 0, 3, -23], [ 0, 4,  -9], [ 0, 5,  -2],
								[ 1,-5,  -1], [ 1,-4,  -8], [ 1,-3, -22], [ 1,-2, -14], [ 1,-1,  52], [ 1, 0, 103], [ 1, 1,  53], [ 1, 2, -14], [ 1, 3, -22], [ 1, 4,  -8], [ 1, 5,  -1],
								[ 2,-5,  -1], [ 2,-4,  -4], [ 2,-3, -15], [ 2,-2, -24], [ 2,-1, -14], [ 2, 0,  -1], [ 2, 1, -14], [ 2, 2, -24], [ 2, 3, -15], [ 2, 4,  -4], [ 2, 5,  -1],						
											  [ 3,-4,  -2], [ 3,-3,  -7], [ 3,-2, -15], [ 3,-1, -22], [ 3, 0, -23], [ 3, 1, -22], [ 3, 2, -15], [ 3, 3,  -7], [ 3, 4,  -2],
															[ 4,-3,  -2], [ 4,-2,  -4], [ 4,-1,  -8], [ 4, 0,  -9], [ 4, 1,  -8], [ 4, 2,  -4], [ 4, 3,  -2],
																		  [ 5,-2,  -1], [ 5,-1,  -1], [ 5, 0,  -2], [ 5, 1,  -1], [ 5, 2,  -1]
								]

	

	sobel_threshold = 38
	sobel_kernel_s1 =	[
							[-1,-1,-1], [ 0,-1,-2], [ 1,-1,-1],
							[-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
							[-1, 1, 1], [ 0, 1, 2], [ 1, 1, 1],
						]
	sobel_kernel_s2 =	[
							[-1,-1,-1], [ 0,-1, 0], [ 1,-1, 1],
							[-1, 0,-2], [ 0, 0, 0], [ 1, 0, 2],
							[-1, 1,-1], [ 0, 1, 0], [ 1, 1, 1],
						]


	
	print ('High Quality Laplacian Processing...')
	lena_laplacian_h = edge_detector(img, laplacian_high_kernel, laplacian_threshold, laplacian_high_nor)
	cv2.imwrite('lena_laplacian_h.bmp', lena_laplacian_h)
	print ('Done!\n')

	print ('Low Quality Laplacian Processing...')
	lena_laplacian_l = edge_detector(img, laplacian_low_kernel, laplacian_threshold)
	cv2.imwrite('lena_laplacian_l.bmp', lena_laplacian_l)
	print ('Done!\n')

	print ('Minimum variance Laplacian Processing...')
	lena_min_var_laplacian = edge_detector(img, laplacian_min_var_kernel, laplacian_min_var_threshold, laplacian_min_var_nor)
	cv2.imwrite('lena_min_var_laplacian.bmp', lena_min_var_laplacian)
	print ('Done!\n')
	
	print ('Laplace of Gaussian Processing...')
	lena_gaussian_laplacian = edge_detector(img, laplacian_gaussian_kernel, laplacian_gaussian_threshold)
	cv2.imwrite('lena_gaussian_laplacian.bmp', lena_gaussian_laplacian)
	print ('Done!\n')

	print ('All task are finish!')

def edge_detector(img, kernel, threshold, normalizer=1.0):
	img_edge = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tmp_sum = 0
			for [x1, x2, w] in kernel:
				a1 = -i-x1-1 if i+x1<0 else i+x1
				a1 = 2*img.shape[0]-i-x1-1 if i+x1>=img.shape[0] else i+x1
				a2 = -j-x2-1 if j+x2<0 else j+x2
				a2 = 2*img.shape[1]-j-x2-1 if j+x2>=img.shape[1] else j+x2
				tmp_sum += img[a1][a2]*w

			img_edge[i][j] = 0 if normalizer*tmp_sum >= threshold else 255

	return img_edge


if __name__ == '__main__':
    main()
