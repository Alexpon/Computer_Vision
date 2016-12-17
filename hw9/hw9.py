import numpy as np
import cv2 as cv2

def main():
	img = cv2.imread('lena.bmp', 0)

	robert_threshold = 12
	robert_kernel_gx = [
							[0, 0, 1], [1, 0, 0],
							[0, 1, 0], [1, 1,-1]
						]
	robert_kernel_gy = [
							[0, 0, 0], [1, 0, 1],
							[0, 1,-1], [1, 1, 0]
						]

	prewitt_threshold = 24
	prewitt_kernel_p1 = [
							[-1,-1,-1], [ 0,-1,-1], [ 1,-1,-1],
							[-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
							[-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1],
						]
	prewitt_kernel_p2 = [
							[-1,-1,-1], [ 0,-1, 0], [ 1,-1, 1],
							[-1, 0,-1], [ 0, 0, 0], [ 1, 0, 1],
							[-1, 1,-1], [ 0, 1, 0], [ 1, 1, 1],
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

	frei_chen_threshold = 30
	frei_chen_kernel_fc1 = [
							[-1,-1,-1], [ 0,-1,-np.sqrt(2)], [ 1,-1,-1],
							[-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
							[-1, 1, 1], [ 0, 1, np.sqrt(2)], [ 1, 1, 1],
						   ]
	frei_chen_kernel_fc2 = [
							[-1,-1,-1], [ 0,-1, 0], [ 1,-1, 1],
							[-1, 0,-np.sqrt(2)], [ 0, 0, 0], [ 1, 0, np.sqrt(2)],
							[-1, 1,-1], [ 0, 1, 0], [ 1, 1, 1],
						   ]

	kirsch_threshold = 135
	kirsch_kernel_0 =	[
							[-1,-1,-3], [ 0,-1,-3], [ 1,-1, 5],
							[-1, 0,-3], [ 0, 0, 0], [ 1, 0, 5],
							[-1, 1,-3], [ 0, 1,-3], [ 1, 1, 5],
						]
	kirsch_kernel_1 =	[
							[-1,-1,-3], [ 0,-1, 5], [ 1,-1, 5],
							[-1, 0,-3], [ 0, 0, 0], [ 1, 0, 5],
							[-1, 1,-3], [ 0, 1,-3], [ 1, 1,-3],
						]
	kirsch_kernel_2 =	[
							[-1,-1, 5], [ 0,-1, 5], [ 1,-1, 5],
							[-1, 0,-3], [ 0, 0, 0], [ 1, 0,-3],
							[-1, 1,-3], [ 0, 1,-3], [ 1, 1,-3],
						]
	kirsch_kernel_3 =	[
							[-1,-1, 5], [ 0,-1, 5], [ 1,-1,-3],
							[-1, 0, 5], [ 0, 0, 0], [ 1, 0,-3],
							[-1, 1,-3], [ 0, 1,-3], [ 1, 1,-3],
						]
	kirsch_kernel_4 =	[
							[-1,-1, 5], [ 0,-1,-3], [ 1,-1,-3],
							[-1, 0, 5], [ 0, 0, 0], [ 1, 0,-3],
							[-1, 1, 5], [ 0, 1,-3], [ 1, 1,-3],
						]						
	kirsch_kernel_5 =	[
							[-1,-1,-3], [ 0,-1,-3], [ 1,-1,-3],
							[-1, 0, 5], [ 0, 0, 0], [ 1, 0,-3],
							[-1, 1, 5], [ 0, 1, 5], [ 1, 1,-3],
						]
	kirsch_kernel_6 =	[
							[-1,-1,-3], [ 0,-1,-3], [ 1,-1,-3],
							[-1, 0,-3], [ 0, 0, 0], [ 1, 0,-3],
							[-1, 1, 5], [ 0, 1, 5], [ 1, 1, 5],
						]
	kirsch_kernel_7 =	[
							[-1,-1,-3], [ 0,-1,-3], [ 1,-1,-3],
							[-1, 0,-3], [ 0, 0, 0], [ 1, 0, 5],
							[-1, 1,-3], [ 0, 1, 5], [ 1, 1, 5],
						]
	kirsch_kernel_set = [	
							kirsch_kernel_0, kirsch_kernel_1, kirsch_kernel_2, kirsch_kernel_3,
							kirsch_kernel_4, kirsch_kernel_5, kirsch_kernel_6, kirsch_kernel_7
						]

	robinson_threshold = 43
	robinson_kernel_0 =	[
							[-1,-1,-1], [ 0,-1, 0], [ 1,-1, 1],
							[-1, 0,-2], [ 0, 0, 0], [ 1, 0, 2],
							[-1, 1,-1], [ 0, 1, 0], [ 1, 1, 1],
						]
	robinson_kernel_1 =	[
							[-1,-1, 0], [ 0,-1, 1], [ 1,-1, 2],
							[-1, 0,-1], [ 0, 0, 0], [ 1, 0, 1],
							[-1, 1,-2], [ 0, 1,-1], [ 1, 1, 0],
						]
	robinson_kernel_2 =	[
							[-1,-1, 1], [ 0,-1, 2], [ 1,-1, 1],
							[-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
							[-1, 1,-1], [ 0, 1,-2], [ 1, 1,-1],
						]
	robinson_kernel_3 =	[
							[-1,-1, 2], [ 0,-1, 1], [ 1,-1, 0],
							[-1, 0, 1], [ 0, 0, 0], [ 1, 0,-1],
							[-1, 1, 0], [ 0, 1,-1], [ 1, 1,-2],
						]
	robinson_kernel_4 =	[
							[-1,-1, 1], [ 0,-1, 0], [ 1,-1,-1],
							[-1, 0, 2], [ 0, 0, 0], [ 1, 0,-2],
							[-1, 1, 1], [ 0, 1, 0], [ 1, 1,-1],
						]
	robinson_kernel_5 =	[
							[-1,-1, 0], [ 0,-1,-1], [ 1,-1,-2],
							[-1, 0, 1], [ 0, 0, 0], [ 1, 0,-1],
							[-1, 1, 2], [ 0, 1, 1], [ 1, 1, 0],
						]
	robinson_kernel_6 =	[
							[-1,-1,-1], [ 0,-1,-2], [ 1,-1,-1],
							[-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
							[-1, 1, 1], [ 0, 1, 2], [ 1, 1, 1],
						]
	robinson_kernel_7 =	[
							[-1,-1,-2], [ 0,-1,-1], [ 1,-1, 0],
							[-1, 0,-1], [ 0, 0, 0], [ 1, 0, 1],
							[-1, 1, 0], [ 0, 1, 1], [ 1, 1, 2],
						]
	robinson_kernel_set = 	[	
							robinson_kernel_0, robinson_kernel_1, robinson_kernel_2, robinson_kernel_3,
							robinson_kernel_4, robinson_kernel_5, robinson_kernel_6, robinson_kernel_7
							]

	nevatia_threshold = 12500
	nevatia_kernel_0 = 	[
							[-2,-2, 100], [-1,-2, 100], [ 0,-2, 100], [ 1,-2, 100], [2,-2, 100],
							[-2,-1, 100], [-1,-1, 100], [ 0,-1, 100], [ 1,-1, 100], [2,-1, 100],
							[-2, 0,   0], [-1, 0,   0], [ 0, 0,   0], [ 1, 0,   0], [2, 0,   0],
							[-2, 1,-100], [-1, 1,-100], [ 0, 1,-100], [ 1, 1,-100], [2, 1,-100],
							[-2, 2,-100], [-1, 2,-100], [ 0, 2,-100], [ 1, 2,-100], [2, 2,-100],
						]
	nevatia_kernel_1 = 	[
							[-2,-2, 100], [-1,-2, 100], [ 0,-2, 100], [ 1,-2, 100], [2,-2, 100],
							[-2,-1, 100], [-1,-1, 100], [ 0,-1, 100], [ 1,-1,  78], [2,-1, -32],
							[-2, 0, 100], [-1, 0,  92], [ 0, 0,   0], [ 1, 0, -92], [2, 0,-100],
							[-2, 1,  32], [-1, 1, -78], [ 0, 1,-100], [ 1, 1,-100], [2, 1,-100],
							[-2, 2,-100], [-1, 2,-100], [ 0, 2,-100], [ 1, 2,-100], [2, 2,-100],
						]					
	nevatia_kernel_2 = 	[
							[-2,-2, 100], [-1,-2, 100], [ 0,-2, 100], [ 1,-2,  32], [2,-2,-100],
							[-2,-1, 100], [-1,-1, 100], [ 0,-1,  92], [ 1,-1, -78], [2,-1,-100],
							[-2, 0, 100], [-1, 0, 100], [ 0, 0,   0], [ 1, 0,-100], [2, 0,-100],
							[-2, 1, 100], [-1, 1,  78], [ 0, 1, -92], [ 1, 1,-100], [2, 1,-100],
							[-2, 2, 100], [-1, 2, -32], [ 0, 2,-100], [ 1, 2,-100], [2, 2,-100],
						]
	nevatia_kernel_3 = 	[
							[-2,-2,-100], [-1,-2,-100], [ 0,-2,   0], [ 1,-2, 100], [2,-2, 100],
							[-2,-1,-100], [-1,-1,-100], [ 0,-1,   0], [ 1,-1, 100], [2,-1, 100],
							[-2, 0,-100], [-1, 0,-100], [ 0, 0,   0], [ 1, 0, 100], [2, 0, 100],
							[-2, 1,-100], [-1, 1,-100], [ 0, 1,   0], [ 1, 1, 100], [2, 1, 100],
							[-2, 2,-100], [-1, 2,-100], [ 0, 2,   0], [ 1, 2, 100], [2, 2, 100],
						]
	nevatia_kernel_4 = 	[
							[-2,-2,-100], [-1,-2,  32], [ 0,-2, 100], [ 1,-2, 100], [2,-2, 100],
							[-2,-1,-100], [-1,-1, -78], [ 0,-1,  92], [ 1,-1, 100], [2,-1, 100],
							[-2, 0,-100], [-1, 0,-100], [ 0, 0,   0], [ 1, 0, 100], [2, 0, 100],
							[-2, 1,-100], [-1, 1,-100], [ 0, 1, -92], [ 1, 1,  78], [2, 1, 100],
							[-2, 2,-100], [-1, 2,-100], [ 0, 2,-100], [ 1, 2, -32], [2, 2, 100],
						]
	nevatia_kernel_5 = 	[
							[-2,-2, 100], [-1,-2, 100], [ 0,-2, 100], [ 1,-2, 100], [2,-2, 100],
							[-2,-1, -32], [-1,-1,  78], [ 0,-1, 100], [ 1,-1, 100], [2,-1, 100],
							[-2, 0,-100], [-1, 0, -92], [ 0, 0,   0], [ 1, 0,  92], [2, 0, 100],
							[-2, 1,-100], [-1, 1,-100], [ 0, 1,-100], [ 1, 1, -78], [2, 1,  32],
							[-2, 2,-100], [-1, 2,-100], [ 0, 2,-100], [ 1, 2,-100], [2, 2,-100],
						]
	nevatia_kernel_set = 	[
								nevatia_kernel_0, nevatia_kernel_1, nevatia_kernel_2,
								nevatia_kernel_3, nevatia_kernel_4, nevatia_kernel_5
							]			
	
	
	print ('Robert Operator Processing...')
	lena_robert = sqrt_edge_detector(img, robert_kernel_gx, robert_kernel_gy, robert_threshold);
	cv2.imwrite('lena_robert.bmp', lena_robert)
	print ('Done!\n')

	print ('Prewitt Edge Detector Processing...')
	lena_prewitt = sqrt_edge_detector(img, prewitt_kernel_p1, prewitt_kernel_p2, prewitt_threshold);
	cv2.imwrite('lena_prewitt.bmp', lena_prewitt)
	print ('Done!\n')

	print ('Sobel Edge Detector Processing...')
	lena_sobel = sqrt_edge_detector(img, sobel_kernel_s1, sobel_kernel_s2, sobel_threshold);
	cv2.imwrite('lena_sobel.bmp', lena_sobel)
	print ('Done!\n')

	print ('Frei and Chen Edge Detector Processing...')
	lena_frei_chen = sqrt_edge_detector(img, frei_chen_kernel_fc1, frei_chen_kernel_fc2, frei_chen_threshold);
	cv2.imwrite('lena_frei_chen.bmp', lena_frei_chen)
	print ('Done!\n')
	
	print ('Kirsch Edge Detector Processing...')
	lena_kirsch = max_edge_detector(img, kirsch_kernel_set, kirsch_threshold);
	cv2.imwrite('lena_kirsch.bmp', lena_kirsch)
	print ('Done!\n')

	print ('Robinson Edge Detector Processing...')
	lena_robinson = max_edge_detector(img, robinson_kernel_set, robinson_threshold);
	cv2.imwrite('lena_robinson.bmp', lena_robinson)
	print ('Done!\n')

	print ('Nevatia and Babu Edge Detector Processing...')
	lena_nevatia = max_edge_detector(img, nevatia_kernel_set, nevatia_threshold);
	cv2.imwrite('lena_nevatia.bmp', lena_nevatia)
	print ('Done!\n')
	
	print ('All task are finish!')

def sqrt_edge_detector(img, kernel1, kernel2, threshold):
	img_k1 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	img_k2 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	img_edge = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			tmp_x = 0
			tmp_y = 0
			for [x1, x2, w] in kernel1:
				a1 = -i-x1-1 if i+x1<0 else i+x1
				a1 = 2*img.shape[0]-i-x1-1 if i+x1>=img.shape[0] else i+x1
				a2 = -j-x2-1 if j+x2<0 else j+x2
				a2 = 2*img.shape[1]-j-x2-1 if j+x2>=img.shape[1] else j+x2
				tmp_x += img[a1][a2]*w

			img_k1[i][j] = tmp_x

			for [y1, y2, w] in kernel2:
				b1 = -i-y1-1 if i+y1<0 else i+y1
				b1 = 2*img.shape[0]-i-y1-1 if i+y1>=img.shape[0] else i+y1
				b2 = -j-y2-1 if j+y2<0 else j+y2
				b2 = 2*img.shape[1]-j-y2-1 if j+y2>=img.shape[1] else j+y2
				tmp_y += img[b1][b2]*w

			img_k2[i][j] = tmp_y

			mix = int(np.sqrt(tmp_x**2 + tmp_y**2))
			img_edge[i][j] = 0 if mix >= threshold else 255

	return img_edge

def max_edge_detector(img, kernel_set, threshold):
	img_edge = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			k_list = []
			for kernel in kernel_set:
				tmp = 0
				for [x1, x2, w] in kernel:
					a1 = -i-x1-1 if i+x1<0 else i+x1
					a1 = 2*img.shape[0]-i-x1-1 if i+x1>=img.shape[0] else i+x1
					a2 = -j-x2-1 if j+x2<0 else j+x2
					a2 = 2*img.shape[1]-j-x2-1 if j+x2>=img.shape[1] else j+x2
					tmp += img[a1][a2]*w
				k_list.append(tmp)
			img_edge[i][j] = 0 if max(k_list)>=threshold else 255
	return img_edge

if __name__ == '__main__':
    main()
