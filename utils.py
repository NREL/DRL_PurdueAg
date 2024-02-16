import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
	headers_in = ['pHP', 'pMP', 'Bulk_mode', 'Alt_mode', 'Vac_mode', 'Fert_mode']
	headers_env = ['Bulk_rpm_cmd', 'Alt_rpm_cmd', 'Vac_rpm_cmd', 'Fert_rpm_cmd']
	headers_out = ['Bulk_P', 'Alt_P', 'Vac_P', 'Fert_P', 
				   'Bulk_Q', 'Alt_Q', 'Vac_Q', 'Fert_Q',
				   'Bulk_rpm_delta', 'Alt_rpm_delta', 'Vac_rpm_delta', 'Fert_rpm_delta']

	headers = headers_in + headers_env + headers_out
	T, data = [], {h: [] for h in headers}
	with h5py.File(filename, 'r') as hf:
		for i in hf:
			if len(list(hf[i]['T'][()])) > len(T):
				T = list(hf[i]['T'][()])

			for h in headers:
				data[h].append(list(hf[i][h][()]))

	T = np.array(T)
	for h in headers:
		data[h] = np.array(data[h])
	
	return T, data

def array_to_dict(X, headers):
	assert X.shape[-1] == len(headers), 'Number of headers must match data dimension'

	X_dict = {}
	for i, h in enumerate(headers):
		X_dict[h] = X[..., i]

	return X_dict

def dict_to_array(X, headers):
	X_array = None
	for h in headers:
		assert h in [_ for _ in X], 'Header not in data dictionary'

		if X_array is None:
			X_array = np.expand_dims(X[h], axis=-1)
		else:
			X_array = np.concatenate((X_array, np.expand_dims(X[h], axis=-1)), axis=-1)

	return X_array

def norm_data(data, scale_factor=None, headers=None, scale_type='minmax'):
	assert (scale_factor is not None) or (headers is not None), 'Must provide scale_factors or headers'

	scale_factor = {} if scale_factor is None else scale_factor

	for i, h in enumerate(headers):
		if h in [_ for _ in scale_factor]:
			sf_type = scale_factor[h][2]
		else:
			sf_type = scale_type
		
		assert sf_type in ['standard', 'minmax'], 'Scaling type invalid.'

		if sf_type == 'standard':
			if h not in [_ for _ in scale_factor]:
				mu, sig = np.mean(data[h], axis=(0, 1)), np.std(data[h], axis=(0, 1))
				sig = 1. if sig == 0. else sig
				scale_factor[h] = [mu, sig, 'standard']

			data[h] = (data[h] - scale_factor[h][0])/scale_factor[h][1]

		elif sf_type == 'minmax':
			if h not in [_ for _ in scale_factor]:
				data_range = np.max(data[h], axis=(0, 1)) - np.min(data[h], axis=(0, 1))
				data_min = np.min(data[h], axis=(0, 1))# - 0.05*data_range
				data_max = np.max(data[h], axis=(0, 1)) + 0.05*data_range
				scale_factor[h] = [data_min, data_max, 'minmax']
				tf = (scale_factor[h][0] == scale_factor[h][1])
				if tf:
					if scale_factor[h][0] != 0.:
						scale_factor[h][0], scale_factor[h][1] = 0., scale_factor[h][1]/2
					else:
						scale_factor[h][0], scale_factor[h][1] = -1., 1.

			#data = 2.*(data - scale_factor[h][0])/(scale_factor[h][1] - scale_factor[h][0]) - 1.
			data[h] = (data[h] - scale_factor[h][0])/(scale_factor[h][1] - scale_factor[h][0])

	return data, scale_factor

def unnorm_data(data, scale_factor, headers):
	for i, h in enumerate(headers):
		sf_h = scale_factor[h][:2]
		sf_type = scale_factor[h][2]
		
		assert sf_type in ['standard', 'minmax'], 'Scaling type invalid.'

		if sf_type == 'standard':
			data[h] = sf_h[1]*data[h] - sf_h[0]
		elif sf_type == 'minmax':
			#data[h] = 0.5*(sf_h[1] - sf_h[0])*(data[h] + 1.) + sf_h[0]
			data[h] = (sf_h[1] - sf_h[0])*data[h] + sf_h[0]

	return data

def save_scale_factors(filename, sf):
	with h5py.File(filename, 'w') as f:
		for h in sf:
			f.create_group(h)
			f[h].create_dataset('sf', data=sf[h][:2])
			f[h].create_dataset('type', data=sf[h][2])

def load_scale_factors(filename):
	sf = {}
	with h5py.File(filename, 'r') as f:
		for h in f:
			f_h = f[h]

			sf_h = list(f_h['sf'][()])
			sf_h.append(f_h['type'][()].decode())

			sf[h] = sf_h
	
	return sf

def build_samples(data, lookback=1, overlap=0.5, headers=None):
	if headers is None:
		headers_in = ['pHP', 'pMP', 'Bulk_mode', 'Alt_mode', 'Vac_mode', 'Fert_mode']
		headers_env = ['Bulk_rpm_cmd', 'Alt_rpm_cmd', 'Vac_rpm_cmd', 'Fert_rpm_cmd']
		headers_out = ['Bulk_P', 'Alt_P', 'Vac_P', 'Fert_P', 
					   'Bulk_Q', 'Alt_Q', 'Vac_Q', 'Fert_Q',
					   'Bulk_rpm_delta', 'Alt_rpm_delta', 'Vac_rpm_delta', 'Fert_rpm_delta']
		headers = {'in': headers_in, 'env': headers_env, 'out': headers_out}

	N_samples = data[headers['in'][0]].shape[0]
	N_steps = data[headers['in'][0]].shape[1]

	if lookback < 0:
		lookback = N_steps-1
		overlap = 0

	X = {h: np.zeros((0, lookback)) for h in headers['in']+headers['env']+headers['out']}
	f = {h: np.zeros((0, lookback)) for h in headers['out']}
	for i in range(N_samples):
		for t in range(0, int(N_steps-lookback), np.maximum(int((1-overlap)*lookback), 1)):
			for h in headers['in']+headers['env']:
				X[h] = np.concatenate((X[h], data[h][i:i+1, t:t+lookback]), axis=0)
			for h in headers['out']:
				X[h] = np.concatenate((X[h], data[h][i:i+1, t:t+lookback]), axis=0)
				f[h] = np.concatenate((f[h], data[h][i:i+1, t+1:t+lookback+1]), axis=0)

	return X, f

def plot_errors(filename, epochs, train_err, test_err):
	plt.figure(figsize=(8, 3))

	plt.subplot(121)
	plt.plot(epochs, train_err, c='r', label='Training RMSE')
	plt.plot(epochs, test_err, c='b', label='Testing RMSE')
	plt.xlim(0, epochs[-1]+1)
	ylim = plt.gca().get_ylim()
	plt.ylim(0, ylim[1])
	plt.legend(loc='upper right')

	plt.subplot(122)
	plt.plot(epochs, train_err, c='r', label='Training RMSE')
	plt.plot(epochs, test_err, c='b', label='Testing RMSE')
	plt.xlim(0, epochs[-1]+1)
	plt.yscale('log')
	plt.legend(loc='upper right')

	plt.tight_layout()
	plt.savefig(filename, dpi=200, bbox_inches='tight')
	plt.close()



