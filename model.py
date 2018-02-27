import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Input, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils
import csv
import keras
import augmentation_engine as aug


def read_log_file(data_path, split_data=False):
	train_data = []
	for path in data_path:
		log_path = path + "/driving_log.csv"
		with open(log_path) as logFile:
			reader = csv.reader(logFile)
			numOfLines = sum(1 for row in reader) - 1
			for row in reader:
				train_data = train_data.append(row)
			
									

	
def read_image_files(data_path, split_data=False):
	X_train = np.zeros((1, 160, 320, 3))
	y_train = np.array([0],dtype='float')
	
	for path in data_path:
		log_path = path + "/driving_log.csv"
		with open(log_path) as logFile:
			reader = csv.reader(logFile)
			numOfLines = sum(1 for row in reader) - 1
			X_train_temp = np.zeros((numOfLines, 160, 320, 3))
			y_train_temp = np.zeros(numOfLines)					

			counter = 0
			logFile.seek(1)
			for row in reader:
				if(counter > 0):
					img_path = row[0]
					if(img_path[0:5] != '/home'):
						img_path = path+'/'+img_path
					print('img_path:' + img_path)
					img = cv2.imread(str(img_path),cv2.IMREAD_COLOR)
					img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					X_train_temp[counter-1] = img
					label = np.array([float(row[3])])
					y_train_temp[counter-1] = label
				counter += 1

		X_train = np.append(X_train,X_train_temp,axis=0)
		y_train = np.append(y_train,y_train_temp,axis=0)
		
	indices = np.where(y_train==0)[0]
	X_train = np.delete(X_train,indices[0:np.floor(0.7*len(indices)).astype('uint')],axis=0).astype(np.uint8)
	y_train = np.delete(y_train,indices[0:np.floor(0.7*len(indices)).astype('uint')],axis=0).astype(np.float32)

	if(split_data):
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
		return X_train, y_train, X_val , y_val
	else:
		return X_train, y_train

def complete_model():

	img_input = Input(shape=[64,64,3])

	x = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block1_conv1')(img_input)
	x = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	x = Convolution2D(128, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block2_conv1')(x)
	x = Convolution2D(128, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	x = Convolution2D(256, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block3_conv1')(x)
	x = Convolution2D(256, 3, 3, init='he_normal', activation='relu', border_mode='same', name='block3_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	x = Flatten(name='my_flatten_1')(x)
	x = Dense(1024, init='he_normal', activation='relu', name='my_fc1')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, init='he_normal', activation='relu', name='my_fc2')(x)
	x = Dropout(0.5)(x)
	x = Dense(64, init='he_normal', activation='relu', name='my_fc3')(x)
	x = Dropout(0.5)(x)
	x = Dense(1, name='predictions')(x)

	model = Model(img_input, x, name='my_model')

	return model

def hyperparameter_selection(model, optimizer, iterations, parameter, val_x, val_y):

	if(parameter == 'learning_rate'):
		temp = np.zeros((iterations,2))
		for i in range(iterations):
			learning_rate = 10**np.random.uniform(-4.,-2., size=None)

			optimizer.lr = learning_rate

			model.compile(loss='mean_squared_error',
				      optimizer=optimizer,
				      metrics=['mean_absolute_error'])
	
			history = model.fit(val_x, val_y, batch_size=32, nb_epoch=2, verbose=1, shuffle=True)
			
			loss = model.evaluate(val_x,val_y)[0]
			temp[i,1] = loss
			temp[i,0] = learning_rate
			
			del history
			
			print("iteration " + str(i) + ":")
			print ("learning_rate: " + str(learning_rate) + ", loss: " +str(loss))

		print("best learning rate at:")
		print(temp[np.argmax(temp[:,1],0),0])
		plt.scatter(temp[i,0],temp[i,1])
		plt.show()

	

def main():

	# Preprocess images
	#'''
	training_path = ["../../training_data/data", "../../training_data/data1"]
	X_train, y_train, X_val, y_val = read_image_files(training_path, split_data=True)
	temp = np.zeros((X_val.shape[0], 64,64,3))
	for i,img in enumerate(X_val):
		temp[i] = aug.preprocess_images(img,64,64)
	del X_val
	X_val = temp
	del temp

	#X_test, y_test = preprocess_data(["../../data"],224,224,split_data=False)
	y_train = y_train*180/np.pi; y_val = y_val*180/np.pi
	#np.save('../X_train.npy',X_train)
	#np.save('../X_val.npy',X_val)
	#np.save('../X_test.npy',X_test)
	#np.save('../y_train.npy',y_train)
	#np.save('../y_val.npy',y_val)
	#np.save('../y_test.npy',y_test)
	#'''

	# Load pre-processed data
	'''
	X_train = np.load('X_train.npy')
	y_train = np.load('y_train.npy')*180/np.pi
	plt.hist(y_train,np.linspace(-90,90,180))
	plt.show()
	indices = np.where(y_train==0)[0]
	print(len(indices))

	X_train = np.delete(X_train,indices[0:np.floor(0.75*len(indices)).astype('uint')],axis=0)
	y_train = np.delete(y_train,indices[0:np.floor(0.75*len(indices)).astype('uint')],axis=0)

	plt.hist(y_train,np.linspace(-90,90,180))
	plt.show()

	print(X_train.shape)
	print(y_train.shape)

	X_val = np.load('X_val.npy')
	y_val = np.load('y_val.npy')*180/np.pi

	#X_test = np.load('X_test.npy')
	#y_test = np.load('y_test.npy')
	'''
	
	# Extract vgg16 Features
	'''
	vgg16_features_train = vgg16_featureExtraction_model(only_model=False, X_train=X_train)
	vgg16_features_val = vgg16_featureExtraction_model(only_model=False, X_train=X_val)
	vgg16_features_test = vgg16_featureExtraction_model(only_model=False, X_train=X_test)

	np.save('vgg16_features_train.npy',vgg16_features_train)
	np.save('vgg16_features_val.npy',vgg16_features_val)
	np.save('vgg16_features_test.npy',vgg16_features_test)
	'''

	# Load saved features
	'''
	vgg16_features_train = np.load('vgg16_features_train.npy')
	vgg16_features_val = np.load('vgg16_features_val.npy')
	vgg16_features_test = np.load('vgg16_features_test.npy')
	'''
	
	# Train the new model on the extracted features
	'''
	batch_size = 256
	final_model = complete_model()
	adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	final_model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['mean_absolute_error'])

	history = final_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=25, validation_data=(X_val,y_val), verbose=1, shuffle=True, 
		callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.2, patience=6, verbose=1, mode='min')])

	final_model.save('model.h5')
	json_string = final_model.to_json()
	text_file = open("model.json", "w")
	text_file.write(json_string)
	text_file.close()
	final_model.summary()
	'''

	# Train the new model on the extracted features
	#'''
	batch_size = 512
	final_model = complete_model()
	
	adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#hyperparameter_selection(final_model, adam, 30, 'learning_rate', X_val, y_val)

	final_model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['mean_absolute_error'])
	train_generator = aug.data_augmentation_engine([X_train, y_train], batch_size)
	final_model.fit_generator(
		    train_generator,
		    samples_per_epoch=20000,
			validation_data=(X_val, y_val),
			callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.2, patience=5, verbose=1, mode='min')],
		    nb_epoch=8)

	final_model.save('model.h5')
	json_string = final_model.to_json()
	text_file = open("model.json", "w")
	text_file.write(json_string)
	text_file.close()
	final_model.summary()
	#'''

	# Test the final model
	#'''
	batch_size = 512
	final_model = complete_model()
	final_model.summary()
	final_model.load_weights('model.h5', by_name=True)
	swa = final_model.predict(X_val,batch_size=batch_size)
	np.save('swa.npy',swa)

	plt.plot(y_val,color='b')
	plt.plot(swa,color='r')
	plt.show()
	#'''

if __name__ == "__main__":
	main()
	


