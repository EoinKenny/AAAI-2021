import alibi
import torch
import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical


def optim_min_edit(cnn, G, z_e, optimizer, target_class):
	"""
	A simple minimal edit based method
	# returns: z prime
	"""

	criterion = torch.nn.CrossEntropyLoss()
	output_t = torch.tensor([target_class], dtype=torch.long)
	epoch = 0

	while True:
		epoch += 1
		optimizer.zero_grad()
		logits, _ = cnn(G(z_e))
		loss = criterion(logits, output_t)
		loss.backward()  
		optimizer.step()  

		pred = int(torch.argmax(logits))

		if int(pred) == target_class:
			return z_e

		if epoch % 5 == 0:
			print(loss.item())


def optim_c_min_edit(G, cnn, x_q, z_e, criterion, optimizer, target_class):
	"""
	A constrained min-edit based on Liu et al. [2019] and Wachter et al. [2017]
	returns: z prime
	"""

	lambda_param = 0.1
	output_t = torch.tensor([target_class], dtype=torch.long)
	epoch = 0

	criterion = torch.nn.CrossEntropyLoss()

	while True:
		epoch += 1
		optimizer.zero_grad()
		output_e, x_e = cnn(G(z_e))  
		loss1 = torch.dist(x_e[0], x_q, 2)
		loss2 = lambda_param * criterion(output_e, output_t)
		loss = loss1 + loss2
		lambda_param += 0.1
		loss.backward(retain_graph=True)  
		optimizer.step()  

		pred = int(torch.argmax(output_e[0]))

		if int(pred) == target_class:
			return z_e

		if epoch % 5 == 0:
			print(loss.item())


def optim_PIECE(G, cnn, x_prime, z_e, criterion, optimizer):
	"""
	Step 3 of the PIECE algorithm
	returns: z prime
	"""

	for i in range(300):

		optimizer.zero_grad()
		logits, x_e = cnn(G(z_e))
		loss = criterion(x_e[0], x_prime)

		loss.backward()  
		optimizer.step()  

		if i % 50 == 0:
			print("Loss:", loss.item())

	return z_e


def optim_Proto_Explanation(query_idx):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255
	x_train = np.reshape(x_train, x_train.shape + (1,))
	x_test = np.reshape(x_test, x_test.shape + (1,))
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	xmin, xmax = -0.5, 0.5
	x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
	x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

	X = x_test[query_idx].reshape((1,) + x_test[query_idx].shape)

	cnn_proto = load_model('weights/keras_cnn.h5', compile=False)
	ae_proto = load_model('weights/mnist_ae.h5', compile=False)
	enc_proto = load_model('weights/mnist_enc.h5', compile=False)

	shape = (1,) + x_train.shape[1:]
	gamma = 100.
	theta = 100.
	c_init = 1.
	c_steps = 2
	max_iterations = 1000
	feature_range = (x_train.min(),x_train.max())

	start = time.time()

	cf = alibi.explainers.CounterFactualProto(cnn_proto, shape, gamma=gamma, theta=theta,
	                         ae_model=ae_proto, enc_model=enc_proto, max_iterations=max_iterations,
	                         feature_range=feature_range, c_init=c_init, c_steps=c_steps)
	cf.fit(x_train)  
	explanation = cf.explain(X)
	print("Took a total of:", time.time() - start)

	try:
		return explanation.cf['X'].reshape(28, 28)
	except:
		return None


def optim_CEM_Explanation(query_idx):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255
	x_train = np.reshape(x_train, x_train.shape + (1,))
	x_test = np.reshape(x_test, x_test.shape + (1,))
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	xmin, xmax = -0.5, 0.5
	x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
	x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

	X = x_test[query_idx].reshape((1,) + x_test[query_idx].shape)
	
	cnn_cem = load_model('weights/keras_cnn.h5', compile=False)
	ae_cem = load_model('weights/mnist_ae.h5', compile=False)

	mode = 'PN'  
	shape = (1,) + x_train.shape[1:]  
	kappa = 0. 
	beta = .1  
	gamma = 100  
	c_init = 1.  
	c_steps = 10  
	max_iterations = 1000  
	feature_range = (x_train.min(),x_train.max())
	clip = (-1000.,1000.) 
	lr = 1e-2  
	no_info_val = -1. 

	start = time.time()
	
	cem = alibi.explainers.CEM(cnn_cem, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
			  gamma=gamma, ae_model=ae_cem, max_iterations=max_iterations,
			  c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

	explanation = cem.explain(X)

	print("Took a total of:", time.time() - start)
	
	try:
		return explanation.PN.reshape(28, 28)
	except:
		return None

		
		