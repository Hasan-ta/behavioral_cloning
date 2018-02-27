from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

cols = 320; rows = 160

def augment_brightness_camera_images(image):
	#print(type(image))
	#cv2.imshow('image',image)
	#cv2.waitKey(0)
	image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	random_bright = .25+np.random.uniform()
	#print(random_bright)
	image1[:,:,2] = image1[:,:,2]*random_bright
	image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
	return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = ((steer/180*np.pi) + tr_x/trans_range*2*.2)*180/np.pi
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

def add_random_shadow(image):
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	#random_bright = .25+.7*np.random.uniform()
	if np.random.randint(2)==1:
		random_bright = .5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
		    image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
		    image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
	return image


def preprocess_images(X, img_length, img_width):
	img = X[55:-1,:,:]
	img1 = cv2.resize(img,dsize=(img_length,img_width)).astype(np.float32)
	img1[:,:,0] -= 103.939
	img1[:,:,1] -= 116.779
	img1[:,:,2] -= 123.68
	return img1

def preprocess_image_file_train(raw_img,swa):
	image,y_steer = trans_image(raw_img,swa,100)
	image = augment_brightness_camera_images(image)
	image = preprocess_images(image, 64, 64)
	image = np.array(image)
	ind_flip = np.random.randint(2)
	if ind_flip==0:
		image = cv2.flip(image,1)
		y_steer = -y_steer
	return image,y_steer

def data_augmentation_engine(data, batch_size = 32):
	batch_images = np.zeros((batch_size, 64, 64, 3))
	batch_steering = np.zeros(batch_size)
	while True:
		for i_batch in range(batch_size):
			i_line = np.random.randint(data[1].shape[0])
			image, label = data[0][i_line], data[1][i_line]
			x,y = preprocess_image_file_train(image, label)
			batch_images[i_batch] = x
			batch_steering[i_batch] = y
		yield batch_images, batch_steering



