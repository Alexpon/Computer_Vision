import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def binarize(img, threshold):
	lena_binarize = img.copy()
	counter = np.zeros(256)
	for i in range(img.shape[0]):
	    for j in range(img.shape[1]):
	        rgb = lena_binarize[i][j][0]
	        counter[rgb] += 1
	        if rgb < threshold:
	            lena_binarize[i][j] = [0,0,0]
	        else:
	            lena_binarize[i][j] = [255,255,255]
	return lena_binarize, counter

def draw_plot(counter):
	plt.bar(np.arange(256), counter)
	plt.xlim(0,255)
	plt.xlabel("rgb")
	plt.ylabel("count")
	plt.show()


def check_size(cnt, ccmap, row, col):
    cnt_list = np.zeros(cnt+1, dtype=np.int)
    for i in range(row):
        for j in range(col):
            cnt_list[ccmap[i][j]] += 1
    return cnt_list

def connected_component(img, row, col):
    cnt = 1
    ccmap = np.zeros((row, col), dtype=np.int)
##### Top-Down left-right and right-left to check equal value
    for i in range(row):
        for j in range(col):
           if (img[i][j][0]==255):
                if(i!=0 and j!=0):
                    if ccmap[i-1][j]!=0:
                        ccmap[i][j]=ccmap[i-1][j]
                    elif ccmap[i][j-1]!=0:
                        ccmap[i][j]=ccmap[i][j-1]
                    else:
                        ccmap[i][j]=cnt
                        cnt += 1
                elif(i==0 and j!=0):
                    if(ccmap[0][j-1]!=0):
                        ccmap[0][j]=ccmap[0][j-1]
                    else:
                        ccmap[0][j]=cnt
                        cnt += 1 
                elif(i!=0 and j==0):
                    if ccmap[i-1][0]!=0:
                        ccmap[i][0]=ccmap[i-1][0]
                    else:
                        ccmap[i][0]=cnt
                        cnt += 1
                else:
                    ccmap[0][0]=cnt
                    cnt+=1
        for k in range(col-2,-1,-1):
            if ccmap[i][k]!=0 and ccmap[i][k+1]!=0 and ccmap[i][k]>ccmap[i][k+1]:
                ccmap[i][k] = ccmap[i][k+1]
    equal = set()

##### Bottom-Up
    for j in range(col):
        for i in range(row-2,-1,-1):
            if(ccmap[i][j]!=0 and ccmap[i+1][j]!=0 and ccmap[i][j]>ccmap[i+1][j]):
                equal.add((ccmap[i+1][j],ccmap[i][j]))
##### Merge Equivalence
    while (len(equal)>0):
        tmp = equal.pop()
        for i in range(row):
            for j in range(col):
                if ccmap[i][j]==tmp[1]:
                    ccmap[i][j]=tmp[0]
    cnt_statistic = check_size(cnt, ccmap, row, col)
    return ccmap, cnt_statistic

def find_target(cnt_statistic, threshold):
	target = []
	for i in range(1,len(cnt_statistic)):
		if(cnt_statistic[i] > threshold):
			target.append(i)
	return target


def find_tdlr(target, ccmap):
	tp_dn_lf_rt = []
	for i in range(ccmap.shape[0]):
		for j in range(ccmap.shape[1]):
			if(target == ccmap[i][j] ):
				tp_dn_lf_rt.append(i)
				break
		if(target == ccmap[i][j] ):
			break

	for i in range(ccmap.shape[0]-1, -1, -1):
		for j in range(ccmap.shape[1]):
			if(target == ccmap[i][j] ):
				tp_dn_lf_rt.append(i)
				break
		if(target == ccmap[i][j] ):
			break

	for j in range(ccmap.shape[1]):
		for i in range(ccmap.shape[0]):
			if(target == ccmap[i][j] ):
				tp_dn_lf_rt.append(j)
				break
		if(target == ccmap[i][j] ):
			break

	for j in range(ccmap.shape[1]-1, -1, -1):
		for i in range(ccmap.shape[0]):
			if(target == ccmap[i][j] ):
				tp_dn_lf_rt.append(j)
				break
		if(target == ccmap[i][j] ):
			break

	return tp_dn_lf_rt


def calculate_centroid(ccmap, tg, tdlr):
	x_sum = 0
	y_sum = 0
	cnt = 0
	for i in range(tdlr[0],tdlr[1]+1):
		for j in range(tdlr[2], tdlr[3]+1):
			if tg==ccmap[i][j]:
				x_sum += j
				y_sum += i
				cnt += 1
	return x_sum/cnt, y_sum/cnt

img = cv2.imread('lena.bmp')
threshold = 128
lena_binarize, counter = binarize(img, threshold)
draw_plot(counter)

ccmap, cnt_statistic = connected_component(lena_binarize, img.shape[0], img.shape[1])
target = find_target(cnt_statistic, 500)

for tg in target:
	tdlr = find_tdlr(tg, ccmap)
	cv2.rectangle(lena_binarize,(tdlr[2],tdlr[0]),(tdlr[3],tdlr[1]),(0,255,0),1)
	x,y = calculate_centroid(ccmap, tg, tdlr)
	cv2.circle(lena_binarize,(x,y), 3, (255,0,0), -1)
	
cv2.imwrite('lena_binarize.bmp',lena_binarize)
cv2.imshow('image', lena_binarize)
cv2.waitKey(0)
cv2.destroyAllWindows()
