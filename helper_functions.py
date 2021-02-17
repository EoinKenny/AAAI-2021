import os
import torch
import torchvision
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import pylab
import numpy as np
import scipy
import torch.optim as optim
import pandas as pd
import time
import tensorflow as tf
import alibi

from IPython.display import clear_output
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from copy import deepcopy
from skimage.io import imread
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

from piece_hurdle_model import *
from local_models import *


def convert_normalization(image):
	temp = deepcopy(image)
	temp = temp * 0.5 
	temp = temp + 0.5
	return temp


def load_models(CNN, Generator):
	G = Generator(ngpu=1)
	cnn = CNN()
	G.load_state_dict  (torch.load('weights/generator.pth',   map_location='cpu'))
	cnn.load_state_dict(torch.load('weights/pytorch_cnn.pth', map_location='cpu'))
	G.eval()
	cnn.eval()
	return G, cnn


def load_autoencoders():
	aes = list()
	for i in range(10):
		model = AE()
		model.load_state_dict(torch.load('weights/ae_' + str(i) + '.pth', map_location='cpu'))
		model.eval()
		aes.append(model)
	ae_full = AE()
	ae_full.load_state_dict(torch.load('weights/ae_full.pth', map_location='cpu'))
	ae_full.eval()
	return aes, ae_full


def get_MNIST_data(datasets):
	mnist_trainset = datasets.MNIST(root='./data/mnist_train', train=True, download=True, transform=None)
	mnist_testset = datasets.MNIST(root ='./data/mnist_test', train=False, download=True, transform=None)
	X_train = mnist_trainset.data
	y_train = mnist_trainset.targets
	X_test = mnist_testset.data
	y_test = mnist_testset.targets
	return X_train, y_train, X_test, y_test


def load_dataloaders():
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5,), (0.5,))])
	train_set = torchvision.datasets.MNIST(
		root='./data/after_anon_review...',
		train=True,
		download=True,
		transform=transform)
	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=1,
		shuffle=False)
	test_set = torchvision.datasets.MNIST(
		root='./data/after_anon_review...',
		train=False,
		download=True,
		transform=transform)
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=1,
		shuffle=False)
	return train_loader, test_loader


def get_missclassificaiton(test_loader, cnn, random_number):
	count = 0
	for i, data in enumerate(test_loader):
		image, label = data
		if label.detach().numpy()[0] != torch.argmax(cnn(image)[0]).detach().numpy():
			count += 1
			if count == random_number: 
				break
	original_query_idx = i
	original_query_img = image
	original_query_label = label.detach().numpy()[0]
	numpy_img = image.reshape(28,28).detach().numpy()
	plt.imshow(convert_normalization(numpy_img))
	plt.axis('off')
	print("Label:", int(label))
	print("Prediction:", torch.argmax(cnn(image)[0]).detach().numpy())
	return original_query_idx, original_query_img, original_query_label

		
def get_data_for_feature(dist_data, target_class, feature_map_num):
	data = np.array(dist_data[target_class])
	data = data.T[feature_map_num].T.reshape(data.shape[0],1)
	return data


def get_distribution_name(dist):
	if dist.fixed_location == True:
		return dist.rv.name + " With Fixed 0 Location"
	else:
		return dist.rv.name

	
def acquire_feature_probabilities(dist_data, target_class, cnn, original_query_img=None, alpha=0.05):
	query_features = cnn(original_query_img)[1][0]
	digit_weights = cnn.classifier[0].weight[target_class]

	# with open('data/pickles/pred_features.pickle', 'rb') as handle:
	# 	dist_data = pickle.load(handle)

	fail_results = list()
	succeed_results = list()
	high_results = list()
	low_results = list()
	expected_values = list()
	probability = list()
	p_values = list()
	distribution_type = list()

	for i in range(len(query_features)):

		data = get_data_for_feature(dist_data, target_class, feature_map_num=i)
		data = data.T[0].T
		feature_value = float(query_features[i])

		dist_examine = HurdleModel(data, value=feature_value, p_value=alpha)
		fail_results.append(dist_examine.bern_fail_sig())  
		succeed_results.append(dist_examine.bern_success_sig())   
		high_results.append(dist_examine.high_cont_sig())  
		low_results.append(dist_examine.low_cont_sig())  
		expected_values.append(dist_examine.get_expected_value())  
		probability.append(dist_examine.get_prob_of_value())
		p_values.append(dist_examine.test_fit())
		distribution_type.append(get_distribution_name(dist_examine))

	df = pd.DataFrame()
	df['Feature Map'] = list(range(len(query_features)))
	df['Contribution'] = query_features.detach().numpy() * digit_weights.detach().numpy()
	df['Bern Fail'] = fail_results
	df['Bern Success'] = succeed_results
	df['Cont High'] = high_results
	df['Cont Low'] = low_results
	df['Expected Value'] = expected_values
	df['Probability of Event'] = probability
	df['Distribtuion p-value KsTest'] = p_values
	df['Dist Type'] = distribution_type

	pd.set_option('display.float_format', lambda x: '%.4f' % x)
	return df


def save_query_and_gan_xp_for_final_data(I_e, cnn, z, G, z_e, original_query_image, name, rand_num):
	numpy_org_image = original_query_image.detach().numpy().reshape(28,28)
	f, axarr = plt.subplots(1,3)
	axarr[0].imshow(numpy_org_image)
	axarr[0].axis('off')
	axarr[0].title.set_text('Query')
	axarr[1].imshow(G(z).detach().numpy().reshape(28,28))
	axarr[1].axis('off')
	axarr[1].title.set_text('GAN Estimation')
	axarr[2].imshow(I_e.detach().numpy().reshape(28,28))
	axarr[2].axis('off')
	axarr[2].title.set_text('Explanation')
	plt.savefig('Explanations/' + name + "_" + str(rand_num) + '.pdf')


def modifying_exceptional_features(df, target_class, query_activations):
	"""
	Change all exceptional features to the expected value for each PDF
	return: tensor with all exceptional features turned into "expected" feature values for c'
	"""

	ideal_xp = query_activations.clone().detach()

	for idx, row in df.sort_values('Probability of Event', ascending=True).iterrows():  # from least probable feature to most probable
		feature_idx = int(row['Feature Map'])  
		expected_value = row['Expected Value'] 
		ideal_xp[feature_idx] = expected_value
	return ideal_xp


def filter_df_of_exceptional_noise(df, target_class, cnn, alpha=0.05):
	"""
	Take the DataFrame, and remove rows which are exceptional features in c' (counterfactual class) but not candidate for change.
	return: dataframe with only relevant features for PIECE algorithm

	alpha is the probability threshold for what is "excetional" or "weird" in the image.
	"""

	df = df[df['Probability of Event'] < alpha]
	df['flag'] = np.zeros(df.shape[0])
	digit_weights = cnn.classifier[0].weight[target_class]

	for idx, row in df.iterrows():
		feature_idx = int(row['Feature Map'])  
		cont = row['Contribution'] 
		cont_high = row['Cont High']
		cont_low = row['Cont Low'] 
		bern_fail = row['Bern Fail']
		expected_value = row['Expected Value']

		if bern_fail:  # if it's unusual to not activate, but it's negative
			if digit_weights[feature_idx] < 0: 
				df.at[feature_idx, 'flag'] = 1
		if cont_high:  # if it's high, but positive
			if digit_weights[feature_idx] > 0: 
				df.at[feature_idx, 'flag'] = 1
		if cont_low:  # if it's low, but negative
			if digit_weights[feature_idx] < 0: 
				df.at[feature_idx, 'flag'] = 1

	df = df[df.flag == 0]
	del df['flag']
	
	return df

