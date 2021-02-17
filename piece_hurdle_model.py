import matplotlib.pyplot as plt
import pylab
import numpy as np
import scipy
import torch.optim as optim
import pandas as pd
import time

from copy import deepcopy

from sklearn.neighbors import KernelDensity

from scipy.spatial.distance import euclidean


class HurdleModel():
	"""
	A statistical hurdle model to find the probability of extracted features at layer X in a CNN for the counterfactual class c'
	"""
	
	def __init__(self, data, value, p_value):
		
		if data.sum() < 0:
			print("*** ERROR: Must be Positive Data ***")

		self.data = data  
		self.value = value 
		self.filtered_data = self.data[self.data != 0]
		self.rv, fixed_location = self.__get_dist_type()

		# Try all six PDF options 
		if fixed_location:
			self.params = self.rv.fit(self.filtered_data, floc=0)
		else:
			self.params = self.rv.fit(self.filtered_data)

		self.fixed_location = fixed_location
		self.p_value = p_value
		self.bern_param = len(self.filtered_data) / len(self.data)  # probability of "success" in Bernoulli trial

	def __get_dist_type(self):

		p_values = {'norm none': None, 'gamma none': None, 'expon none': None,
					'norm floc': None, 'gamma floc': None, 'expon floc': None}

		for test in ['norm', 'gamma', 'expon']:
			for location in ['none', 'floc']:
				if test == 'norm':
					rv = scipy.stats.norm
				elif test == 'gamma':    
					rv = scipy.stats.gamma
				elif test == 'expon':    
					rv = scipy.stats.expon

				if location == 'none':
					params = rv.fit(self.filtered_data) 
				elif location == 'floc':
					params = rv.fit(self.filtered_data, floc=0)

				p_values[test + " " + location] = (scipy.stats.kstest(self.filtered_data, test, args=params)[1])

		max_key = max(p_values, key=lambda k: p_values[k])
		dist_type, location = max_key.split(" ")

		if dist_type == 'norm':
			if location == 'none':
				return scipy.stats.norm, False
			elif location == 'floc':
				return scipy.stats.norm, True
		if dist_type == 'gamma':
			if location == 'none':
				return scipy.stats.gamma, False
			elif location == 'floc':
				return scipy.stats.gamma, True
		if dist_type == 'expon':
			if location == 'none':
				return scipy.stats.expon, False
			elif location == 'floc':
				return scipy.stats.expon, True

	def __pdf(self, x):
		return self.rv.pdf(x, *self.params) * self.bern_param

	def __cdf(self, x):
		return self.rv.cdf(x, *self.params) * self.bern_param

	def get_cdf(self, x):
		return self.rv.cdf(x, *self.params) * self.bern_param

	def __ppf_upper_sig_value(self):
		return self.rv.ppf(0.999, *self.params) * self.bern_param

	def __ppf_lower_sig_value(self):
		return self.rv.ppf(0.001, *self.params) * self.bern_param

	def get_expected_value(self):
		return self.rv.mean(*self.params)

	def get_prob_of_value(self):
		if self.value == 0:
			return 1 - self.bern_param
		else:
			lower = self.__cdf(self.value)
			upper = self.bern_param - self.__cdf(self.value)
			return min(lower, upper)

	def bern_fail_sig(self):
		if (1 - self.bern_param) < self.p_value and self.value == 0:
			return True
		return False

	def bern_success_sig(self):
		if self.bern_param < (self.p_value*2) and self.value > 0:
			return True
		return False

	def high_cont_sig(self):
		if (1 - self.bern_param) + self.__cdf(self.value) > (1-self.p_value):
			return True
		return False

	def low_cont_sig(self):
		if self.__cdf(self.value) < self.p_value and self.value != 0:
			return True
		return False

	def test_fit(self):
		return scipy.stats.kstest(self.filtered_data, self.rv.name, args=self.params)[1]

