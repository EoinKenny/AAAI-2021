import numpy as np
import torch



def IM1_metric(explanation_img, aes, original_class, target_class):
	"""
	return: IM1 metric
	"""

	explanation_img = (explanation_img * 0.5) + 0.5
	o_recon = aes[original_class](explanation_img.reshape(-1,1,28,28)).flatten()
	e_recon = aes[target_class](explanation_img.reshape(-1,1,28,28)).flatten()
	o_error = sum(   (  o_recon.detach().numpy().flatten() - explanation_img.detach().numpy().flatten() )**2   )  
	e_error = sum(   (  e_recon.detach().numpy().flatten() - explanation_img.detach().numpy().flatten() )**2   )  
	return e_error / o_error


def IM2_metric(explanation_img, aes, ae_full, target_class):
	"""
	return: IM2 metric
	"""

	explanation_img = (explanation_img * 0.5) + 0.5
	all_recon = ae_full(explanation_img.reshape(-1,1,28,28)).flatten().detach().numpy()
	e_recon = aes[target_class](explanation_img.reshape(-1,1,28,28)).flatten().detach().numpy()
	x_l1_norm = float(sum(abs(explanation_img.flatten())))
	return sum(  (e_recon - all_recon)**2  ) / x_l1_norm


def mc_dropout(cnn, new_pred, I_e):
	"""
	return: posterior distribution using Monte Carlo Dropout in numpy format
	"""

	cnn = cnn.train()
	for m in cnn.modules():
		if isinstance(m, torch.nn.BatchNorm2d):
			m.eval()
	mc_dropout_results = list()
	for _ in range(1000):  # num of forward passes
		pred = float(torch.exp(  cnn(I_e)[0]  )[0][new_pred])
		mc_dropout_results.append(pred)
	mc_dropout_results = np.array(mc_dropout_results)
	cnn = cnn.eval()
	return mc_dropout_results


def substitutability():
	"""
	How well does the generated CFs replace the actual training data in a k-NN fit to pixel space?
	"""

	return 0


def nn_dist_evaluation():
	"""
	how close is the explanation to an actual training datapoint? (i.e., a "possible world"[Wachter et al. 2016])
	"""

	return 0



