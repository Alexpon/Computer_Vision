import numpy as np
import cv2 as cv2


def main():
	# [i, j, w] : [row, col, weight]
	box33 = [
				[-1,-1, 1], [ 0,-1, 1], [ 1,-1, 1],
				[-1, 0, 1], [ 0, 0, 1], [ 1, 0, 1],
				[-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1]	
			]
	box55 = [
				[-2,-2, 1], [-1,-2, 1], [ 0,-2, 1], [ 1,-2, 1], [ 2,-2, 1],
				[-2,-1, 1], [-1,-1, 1], [ 0,-1, 1], [ 1,-1, 1], [ 2,-1, 1],
				[-2, 0, 1], [-1, 0, 1], [ 0, 0, 1], [ 1, 0, 1], [ 2, 0, 1],
				[-2, 1, 1], [-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1], [ 2, 1, 1],	
				[-2, 2, 1], [-1, 2, 1], [ 0, 2, 1], [ 1, 2, 1], [ 2, 2, 1]	
			]

	kernel =[			  [-2, -1], [-2, 0], [-2, 1],
       			[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        		[ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
        		[ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2],
        				  [ 2, -1], [ 2, 0], [ 2, 1]
    		]

	img = cv2.imread('lena.bmp', 0)

	print ('Noise Generating Start:')
	print ('Gaussian Noise Generating...')
	lena_gaussian10 = gaussian_noise(img, 10)
	cv2.imwrite('lena_gaussian10.bmp', lena_gaussian10)
	lena_gaussian30 = gaussian_noise(img, 30)
	cv2.imwrite('lena_gaussian30.bmp', lena_gaussian30)

	print ('Salt and Pepper Noise Generating...')
	lena_s_p_005 = salt_and_pepper(img, 0.05)
	cv2.imwrite('lena_salt_pepper_005.bmp', lena_s_p_005)
	lena_s_p_010 = salt_and_pepper(img, 0.1)
	cv2.imwrite('lena_salt_pepper_010.bmp', lena_s_p_010)
	print ('Finish Noise Generating\n')

	print ('Box Filter Processing Start:')
	print ('3*3 Box Filter processing on lena_gaussian10...')
	lena_box33_ga10 = box_filter(lena_gaussian10, box33)
	cv2.imwrite('lena_box33_ga10.bmp', lena_box33_ga10)

	print ('3*3 Box Filter processing on lena_gaussian30...')
	lena_box33_ga30 = box_filter(lena_gaussian30, box33)
	cv2.imwrite('lena_box33_ga30.bmp', lena_box33_ga30)

	print ('3*3 Box Filter processing on lena_salt_pepper_005...')
	lena_box33_sp005 = box_filter(lena_s_p_005, box33)
	cv2.imwrite('lena_box33_sp005.bmp', lena_box33_sp005)

	print ('3*3 Box Filter processing on lena_salt_pepper_010...')
	lena_box33_sp010 = box_filter(lena_s_p_010, box33)
	cv2.imwrite('lena_box33_sp010.bmp', lena_box33_sp010)

	print ('5*5 Box Filter processing on lena_gaussian10...')
	lena_box55_ga10 = box_filter(lena_gaussian10, box55)
	cv2.imwrite('lena_box55_ga10.bmp', lena_box55_ga10)

	print ('5*5 Box Filter processing on lena_gaussian30...')
	lena_box55_ga30 = box_filter(lena_gaussian30, box55)
	cv2.imwrite('lena_box55_ga30.bmp', lena_box55_ga30)

	print ('5*5 Box Filter processing on lena_salt_pepper_005...')
	lena_box55_sp005 = box_filter(lena_s_p_005, box55)
	cv2.imwrite('lena_box55_sp005.bmp', lena_box55_sp005)

	print ('5*5 Box Filter processing on lena_salt_pepper_010...')
	lena_box55_sp010 = box_filter(lena_s_p_010, box55)
	cv2.imwrite('lena_box55_sp010.bmp', lena_box55_sp010)
	print ('Finish Box Filter Processing\n')

	print ('Median Filter Processing Start:')
	print ('3*3 Median Filter processing on lena_gaussian10...')
	lena_med33_ga10 = median_filter(lena_gaussian10, box33)
	cv2.imwrite('lena_med33_ga10.bmp', lena_med33_ga10)

	print ('3*3 Median Filter processing on lena_gaussian30...')
	lena_med33_ga30 = median_filter(lena_gaussian30, box33)
	cv2.imwrite('lena_med33_ga30.bmp', lena_med33_ga30)

	print ('3*3 Median Filter processing on lena_salt_pepper_005...')
	lena_med33_sp005 = median_filter(lena_s_p_005, box33)
	cv2.imwrite('lena_med33_sp005.bmp', lena_med33_sp005)
	
	print ('3*3 Median Filter processing on lena_salt_pepper_010...')
	lena_med33_sp010 = median_filter(lena_s_p_010, box33)
	cv2.imwrite('lena_med33_sp010.bmp', lena_med33_sp010)

	print ('5*5 Median Filter processing on lena_gaussian10...')
	lena_med55_ga10 = median_filter(lena_gaussian10, box55)
	cv2.imwrite('lena_med55_ga10.bmp', lena_med55_ga10)

	print ('5*5 Median Filter processing on lena_gaussian30...')
	lena_med55_ga30 = median_filter(lena_gaussian30, box55)
	cv2.imwrite('lena_med55_ga30.bmp', lena_med55_ga30)

	print ('5*5 Median Filter processing on lena_salt_pepper_005...')
	lena_med55_sp005 = median_filter(lena_s_p_005, box55)
	cv2.imwrite('lena_med55_sp005.bmp', lena_med55_sp005)
	
	print ('5*5 Median Filter processing on lena_salt_pepper_010...')
	lena_med55_sp010 = median_filter(lena_s_p_010, box55)
	cv2.imwrite('lena_med55_sp010.bmp', lena_med55_sp010)
	print ('Finish Median Filter Processing\n')

	print ('Closing-Opening Processing Start:')
	print ('Closing-Opening processing on lena_gaussian10...')
	lena_cl_op_ga10 = opening(closing(lena_gaussian10, kernel), kernel)
	cv2.imwrite('lena_cl_op_ga10.bmp', lena_cl_op_ga10)
	
	print ('Closing-Opening processing on lena_gaussian30...')
	lena_cl_op_ga30 = opening(closing(lena_gaussian30, kernel), kernel)
	cv2.imwrite('lena_cl_op_ga30.bmp', lena_cl_op_ga30)
	
	print ('Closing-Opening processing on lena_salt_pepper_005...')
	lena_cl_op_sp005 = opening(closing(lena_s_p_005, kernel), kernel)
	cv2.imwrite('lena_cl_op_sp005.bmp', lena_cl_op_sp005)

	print ('Closing-Opening processing on lena_salt_pepper_010...')
	lena_cl_op_sp010 = opening(closing(lena_s_p_010, kernel), kernel)
	cv2.imwrite('lena_cl_op_sp010.bmp', lena_cl_op_sp010)
	print ('Finish Closing-Opening processing')

	print ('Opening-Closing Processing Start:')
	print ('Opening-Closing processing on lena_gaussian10...')
	lena_op_cl_ga10 = closing(opening(lena_gaussian10, kernel), kernel)
	cv2.imwrite('lena_op_cl_ga10.bmp', lena_op_cl_ga10)
	
	print ('Opening-Closing processing on lena_gaussian30...')
	lena_op_cl_ga30 = closing(opening(lena_gaussian30, kernel), kernel)
	cv2.imwrite('lena_op_cl_ga30.bmp', lena_op_cl_ga30)
	
	print ('Opening-Closing processing on lena_salt_pepper_005...')
	lena_op_cl_sp005 = closing(opening(lena_s_p_005, kernel), kernel)
	cv2.imwrite('lena_op_cl_sp005.bmp', lena_op_cl_sp005)

	print ('Opening-Closing processing on lena_salt_pepper_010...')
	lena_op_cl_sp010 = closing(opening(lena_s_p_010, kernel), kernel)
	cv2.imwrite('lena_op_cl_sp010.bmp', lena_op_cl_sp010)
	print ('Finish All Task')


def gaussian_noise(img, amplitude):
	noise_img = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			new_pixel = img[i][j] + amplitude*np.random.normal()
			noise_img[i][j] = new_pixel if new_pixel < 255 else 255
	return noise_img

def salt_and_pepper(img, threshold):
	noise_img = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			uni = np.random.uniform()
			if uni < threshold:
				noise_img[i][j] = 0
			elif uni > (1-threshold):
				noise_img[i][j] = 255
			else:
				noise_img[i][j] = img[i][j]
	return noise_img

def box_filter(img, box):
	filted_img = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			pixel_val = 0
			normalize = 0
			for [m, n, l] in box:
				if (i+m>=0 and i+m<img.shape[0] and j+n>=0 and j+n<img.shape[1]):
					normalize += l
					pixel_val += filted_img[i+m][j+n]*l
			filted_img[i][j] = pixel_val/normalize
	return filted_img

def median_filter(img, box):
	filted_img = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			counter = 0
			tmp_list = []
			for [m, n, l] in box:
				if (i+m>=0 and i+m<img.shape[0] and j+n>=0 and j+n<img.shape[1]):
					counter += l
					tmp_list.append(img[i+m][j+n])
			tmp_list.sort()
			if (counter % 2 == 1):
				filted_img[i][j] = tmp_list[counter/2]
			else:
				filted_img[i][j] = (tmp_list[counter/2-1]+tmp_list[counter/2])/2
	return filted_img

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

if __name__ == '__main__':
	main()
