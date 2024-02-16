import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
 
class ROM(nn.Module):
	def __init__(self, x_in, f_in, headers=None, scale_factors=None, model_path=None):
		super(ROM, self).__init__()

		dropout_rate = 0.1

		self.x, self.f = x_in, f_in
		self.scale_factors = scale_factors
		self.headers = headers
		if self.headers is not None:
			self.x_headers = headers['in']+headers['env']+headers['out']
			self.f_headers = headers['out']
		else:
			self.x_headers, self.f_headers = None, None

		self.dt = 0.2

		self.lstm0 = nn.LSTM(input_size=self.x, hidden_size=32, num_layers=2, 
							 dropout=dropout_rate, batch_first=True)
		self.linear0 = nn.Linear(32, 32)
		self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=2,
							 dropout=dropout_rate, batch_first=True)
		self.linear1 = nn.Linear(64, 64)
		self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, 
							 dropout=dropout_rate, batch_first=True)
		self.linear2 = nn.Linear(32, 16)
		self.linear3 = nn.Linear(16, self.f)

		self.dropout = nn.Dropout(dropout_rate)

		if model_path is not None:
			self.load_state_dict(torch.load(model_path))
			self.eval()

	def forward(self, x):

		x, _ = self.lstm0(x)
		x = self.dropout(x)
		x = nn.Sigmoid()(self.linear0(x))

		x, _ = self.lstm1(x)
		x = self.dropout(x)
		x = nn.Sigmoid()(self.linear1(x))

		x, _ = self.lstm2(x)
		x = self.dropout(x)
		x = nn.Sigmoid()(self.linear2(x))
		x = self.dropout(x)
		x = nn.Sigmoid()(self.linear3(x))

		return x

	def plot_timeseries(self, loader=None, pathname='figs/'):
		path_str = ''
		for dir_name in pathname.split('/'):
			path_str += dir_name+'/'
			if not os.path.isdir(path_str):
				os.makedirs(path_str)

		self.eval()

		with torch.no_grad():
			count = 0
			for X_batch, f_batch in loader:
				f_pred = self(X_batch)
				
				X_batch = utils.array_to_dict(X_batch.numpy(), self.x_headers)
				f_batch = utils.array_to_dict(f_batch.numpy(), self.f_headers)
				f_pred = utils.array_to_dict(f_pred.numpy(), self.f_headers)
				
				XX_batch = utils.unnorm_data(X_batch, self.scale_factors, self.x_headers)
				ff_batch = utils.unnorm_data(f_batch, self.scale_factors, self.f_headers)
				ff_pred = utils.unnorm_data(f_pred, self.scale_factors, self.f_headers)
				
				XX_batch = utils.dict_to_array(XX_batch, self.x_headers)
				ff_batch = utils.dict_to_array(ff_batch, self.f_headers)
				ff_pred = utils.dict_to_array(ff_pred, self.f_headers)

				for i in range(XX_batch.shape[0]):
					X_i, f_i, f_pred_i = XX_batch[i], ff_batch[i], ff_pred[i]
					T = np.arange(self.dt, (f_i.shape[0]+0.1)*self.dt, self.dt)
					for j, h_j in enumerate(self.f_headers):
						plt.figure(figsize=(5, 3))
						plt.plot(T, f_i[:, j], c='b', linewidth=1, label='True')
						plt.plot(T, f_pred_i[:, j], c='r', ls=':', linewidth=1, label='ROM')
						plt.xlim(0, 300)
						plt.xlabel('Time')
						plt.ylabel(h_j)
						plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
						plt.savefig(pathname+'/{:05d}_output_{}.png'.format(count, h_j), 
									dpi=250, bbox_inches='tight')
						plt.close()

					count += 1

	def call_ROM(self, act, cmd, obs, T=None, normalized=False):
		# action = Dict('discrete': NumPy (4, t), 'continuous': NumPy (2, t))
		#				order: Bulk_mode, Alt_mode, Vac_mode, Fert_mode
		#				order: pHP, pMP
		# cmd_rpm = Dict('continuous': NumPy (4, t))
		#				 order: Bulk_rpm, Alt_rpm, Vac_rpm, Fert_rpm
		# obs = Dict('continuous': NumPy (12, t))
		#			 continuous order: Bulk_P, Alt_P, Vac_P, Fert_P, Bulk_Q, Alt_Q, Vac_Q, Fert_Q,
		#							   Bulk_rpm_delta, Alt_rpm_delta, Vac_rpm_delta, Fert_rpm_delta
		
		T = self.dt if T is None else T
		steps = np.maximum(int(T/self.dt), 1)

		X = np.concatenate((act['continuous'], 
							act['discrete'],
							cmd['continuous'],
							obs['continuous']), axis=0)
		
		if X.ndim == 2:
			ndim_in = 2
			if (X.shape[1] != self.x) and (X.shape[0] == self.x):
				X = X.T
			elif (X.shape[1] != self.x) and (X.shape[0] != self.x):
				assert False, 'X must have {} feature channels'.format(self.x)
			X = np.expand_dims(X, axis=0)
		elif X.ndim == 3:
			ndim_in = 3
			if (X.shape[2] != self.x) and (X.shape[1] == self.x):
				X = np.transpose(X, axes=[0, 2, 1])
			elif (X.shape[2] != self.x) and (X.shape[1] != self.x):
				assert False, 'X must have {} feature channels'.format(self.x)
		else:
			assert False, 'X must be shape [N_timesteps, N_features] or [N_batch, N_timesteps, N_features]'

		X_dict = utils.array_to_dict(X, self.x_headers)
		if not normalized:
			X_dict, _ = utils.norm_data(X_dict, 
								   scale_factor=self.scale_factors, 
								   headers=self.x_headers)
		self.eval()

		for i in range(steps):
			with torch.no_grad():
				X = utils.dict_to_array(X_dict, self.x_headers)
				if X.shape[1] > 100:
					X = X[:, -100:, :]
				f = self(torch.tensor(X, dtype=torch.float32)).numpy()

				f = utils.array_to_dict(f, self.f_headers)

				for h in self.x_headers:
					if h in self.f_headers:
						X_dict[h] = np.concatenate((X_dict[h], f[h][:, -1:]), axis=1)
					else:
						X_dict[h] = np.concatenate((X_dict[h], X_dict[h][:, -1:]), axis=1)

		if not normalized:
			f = utils.unnorm_data(f, self.scale_factors, self.f_headers)
		
		f = np.concatenate((f['Bulk_P'], f['Alt_P'], f['Vac_P'], f['Fert_P'], 
							f['Bulk_Q'], f['Alt_Q'], f['Vac_Q'], f['Fert_Q'], 
							f['Bulk_rpm_delta'], f['Alt_rpm_delta'], f['Vac_rpm_delta'], f['Fert_rpm_delta']), axis=0)
		if not normalized:
			f[f < 0.] = 0.

		return {'continuous': f[:, -1:]}

	def train_model(self, train_loader, test_loader=None, plot_loader=None,
						  N_epochs=100, learning_rate=1e-3, save_every=1, print_every=1,
						  save_model_path=None):
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		loss_fn = nn.MSELoss()

		epoch_errors, training_errors, testing_errors = [], [], []
		for epoch in range(N_epochs):
			if epoch % print_every == 0:
					print('Starting epoch {}'.format(epoch))

			tic = time()
			
			self.train()

			for X_batch, f_batch in train_loader:
				f_pred = self(X_batch)
				loss = loss_fn(f_pred, f_batch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			
			# Print model performance
			if epoch % print_every != 0:
				continue

			print_string = 'Epoch {}'.format(epoch)

			self.eval()
			
			with torch.no_grad():
				epoch_errors.append(epoch+1)
				train_rmse = torch.zeros((0, ))
				for X_batch, f_batch in train_loader:
					f_pred = self(X_batch)
					train_rmse = torch.concat([train_rmse, 
											   torch.mean((f_pred-f_batch)**2, dim=(1, 2))], dim=0)
				train_rmse = torch.sqrt(torch.mean(train_rmse))
				training_errors.append(train_rmse.item())
				print_string += ': train RMSE {:.4f}'.format(train_rmse)

				if test_loader is not None:
					test_rmse = torch.zeros((0, ))
					for X_batch, f_batch in test_loader:
						f_pred = self(X_batch)
						test_rmse = torch.concat([test_rmse, 
												  torch.mean((f_pred-f_batch)**2, dim=(1, 2))], dim=0)
					test_rmse = torch.sqrt(torch.mean(test_rmse))
					print_string += ', test RMSE {:.4f}'.format(test_rmse)
				testing_errors.append(test_rmse.item())
			print(print_string)
			print('Time to complete: {:.02f} s\n'.format(time()-tic))

			# Save model state
			if epoch % save_every != 0:
				continue
			
			if save_model_path is not None:
				model_epoch = save_model_path+'/{0:08d}'.format(epoch)
				if not os.path.exists(model_epoch):
					os.makedirs(model_epoch)
				torch.save(self.state_dict(), '/'.join([model_epoch, 'rom.h5']))
				utils.save_scale_factors('/'.join([model_epoch, 'scale_factors.h5']), self.scale_factors)

				plot_path = save_model_path+'/{0:08d}/errors.png'.format(epoch)
				utils.plot_errors(plot_path, epoch_errors, training_errors, testing_errors)

				plot_path = save_model_path+'/{0:08d}/figs'.format(epoch)
				if plot_loader is not None:
					self.plot_timeseries(loader=plot_loader, pathname=plot_path)

		if save_model_path is not None:
			if not os.path.exists(save_model_path):
				os.makedirs(save_model_path)
			torch.save(self.state_dict(), '/'.join([save_model_path, 'rom.h5']))
			utils.save_scale_factors('/'.join([model_epoch, 'scale_factors.h5']), self.scale_factors)


