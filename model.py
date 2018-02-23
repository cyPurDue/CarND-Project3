import csv
import cv2
import numpy as np

lines = []
with open('data_6/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		#if line[0] == "center":
		#	continue
		lines.append(line)
print('loading lines...')

images = []
measurements = []
steering_factor = 0.2
steering_measure_vec = [0, steering_factor, -steering_factor]
print('loading images...')
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data_6/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3]) + steering_measure_vec[i]
		measurements.append(measurement)

print('augmenting images...')
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print('Done: X_train, y_train')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape = (160, 320, 3)))
#model.add(Dense(1))
model.add(Cropping2D(cropping = ((50, 20), (0, 0))))
model.add(Convolution2D(24,5,5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48,5,5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')
exit()
