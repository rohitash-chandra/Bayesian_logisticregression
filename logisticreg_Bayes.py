 
 # by R. Chandra
 #Source: https://github.com/rohitash-chandra/logistic_regression

import numpy as np
import random
import math

from math import exp

SIGMOID = 1
STEP = 2
LINEAR = 3

 
random.seed()

class logistic_regression:

	def __init__(self, num_epocs, train_data, test_data, num_features, learn_rate):
		self.train_data = train_data
		self.test_data = test_data 
		self.num_features = num_features
		self.num_outputs = self.train_data.shape[1] - num_features 
		self.num_train = self.train_data.shape[0]
		self.w = np.random.uniform(-0.5, 0.5, num_features)  # in case one output class 
		self.b = np.random.uniform(-0.5, 0.5, self.num_outputs) 
		self.learn_rate = learn_rate
		self.max_epoch = num_epocs
		self.use_activation = SIGMOID #SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear 
		self.out_delta = np.zeros(self.num_outputs)
 
	def activation_func(self,z_vec):
		if self.use_activation == SIGMOID:
			y =  1 / (1 + np.exp(z_vec)) # sigmoid/logistic
		elif self.use_activation == STEP:
			y = (z_vec > 0).astype(int) # if greater than 0, use 1, else 0
			#https://stackoverflow.com/questions/32726701/convert-real-valued-numpy-array-to-binary-array-by-sign
		else:
			y = z_vec
		return y
 

	def predict(self, x_vec ): 
		z_vec = x_vec.dot(self.w) - self.b 
		output = self.activation_func(z_vec) # Output  
		return output
	


	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class
 
	def encode(self, w): # get  the parameters and encode into the model
		 
		self.w =  w[0:self.num_features]
		self.b = w[self.num_features] 

	def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

		self.encode(w)  # method to encode w and b
		fx = np.zeros(data.shape[0]) 

		for s in range(0, data.shape[0]):  
				i = random.randint(0, data.shape[0]-1)
				input_instance  =  data[i,0:self.num_features]  
				actual  = data[i,self.num_features:]  
				prediction = self.predict(input_instance)  
				fx[s] = prediction

		return fx

	def gradient(self, x_vec, output, actual):  # not used in case of Random Walk proposals in MCMC 
		if self.use_activation == SIGMOID :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =   (output - actual) 
		return out_delta

	def update(self, x_vec, output, actual):      # not used by MCMC alg
		self.w+= self.learn_rate *( x_vec *  self.out_delta)
		self.b+=  (1 * self.learn_rate * self.out_delta)
 	
 

#------------------------------------------------------------------


class MCMC:
	def __init__(self, samples, traindata, testdata, topology):
		self.samples = samples  # NN topology [input, hidden, output]
		self.topology = topology  # max epocs
		self.traindata = traindata  #
		self.testdata = testdata
		random.seed() 

	def rmse(self, predictions, targets):
		return np.sqrt(((predictions - targets) ** 2).mean())

	def likelihood_func(self, model, data, w, tausq):
		y = data[:, self.topology[0]]
		fx = model.evaluate_proposal(data, w)
		rmse = self.rmse(fx, y)
		loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
		return [np.sum(loss), fx, rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq): 
		param = self.topology[0]  + 1 # number of parameters in model
		part1 = -1 * (param / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss

	def sampler(self):

		# ------------------- initialize MCMC
		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]
		samples = self.samples

		x_test = np.linspace(0, 1, num=testsize)
		x_train = np.linspace(0, 1, num=trainsize)

		#self.topology  # [input,   output]
		y_test = self.testdata[:, self.topology[0]]
		y_train = self.traindata[:, self.topology[0]]
	  
		w_size = self.topology[0]  + self.topology[1]  # num of weights and bias (eg. 4 + 1 in case of time series problems used)

		pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
		pos_tau = np.ones((samples, 1))

		fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
		fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
		#rmse_train = np.zeros(samples)
		#rmse_test = np.zeros(samples)

		w = np.random.randn(w_size)
		w_proposal = np.random.randn(w_size)

		step_w = 0.02;  # defines how much variation you need in changes to w
		step_eta = 0.01;  
		# eta is an additional parameter to cater for noise in predictions (Gaussian likelihood). 
		# note eta is used as tau in the sampler to consider log scale. 
		# eta is not used in multinomial likelihood. 
 

		model = logistic_regression(0 ,  self.traindata, self.testdata, self.topology[0], 0.1) 

		pred_train = model.evaluate_proposal(self.traindata, w)
		pred_test = model.evaluate_proposal(self.testdata, w)

		eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		print('evaluate Initial w')

		sigma_squared = 5  # considered by looking at distribution of  similar trained  models - i.e distribution of weights and bias
		nu_1 = 0
		nu_2 = 0

		prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

		[likelihood, pred_train, rmsetrain] = self.likelihood_func(model, self.traindata, w, tau_pro)

		print(likelihood, ' initial likelihood')
		[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(model, self.testdata, w, tau_pro)


		naccept = 0  

		for i in range(samples - 1):

			w_proposal = w + np.random.normal(0, step_w, w_size)

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = math.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(model, self.traindata, w_proposal, tau_pro)
			[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(model, self.testdata, w_proposal, tau_pro)

			# likelihood_ignore  refers to parameter that will not be used in the alg.

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro)  # takes care of the gradients

			diff_likelihood = likelihood_proposal - likelihood # since we using log scale: based on https://www.rapidtables.com/math/algebra/Logarithm.html
			diff_priorliklihood = prior_prop - prior_likelihood

			mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

			u = random.uniform(0, 1)

			if u < mh_prob:
				# Update position
				print    (i, ' is accepted sample')
				naccept += 1
				likelihood = likelihood_proposal
				prior_likelihood = prior_prop
				w = w_proposal
				eta = eta_pro

				print (likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')

				pos_w[i + 1,] = w_proposal
				pos_tau[i + 1,] = tau_pro
				fxtrain_samples[i + 1,] = pred_train
				fxtest_samples[i + 1,] = pred_test 

			else:
				pos_w[i + 1,] = pos_w[i,]
				pos_tau[i + 1,] = pos_tau[i,]
				fxtrain_samples[i + 1,] = fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,] 

		print(naccept, ' num accepted')
		print(naccept / (samples * 1.0), '% was accepted')
		accept_ratio = naccept / (samples * 1.0) * 100
 

		rmse_train = 0
		rmse_test = 0

		return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, rmse_train, rmse_test, accept_ratio)

 

def main():
  

	outres = open('results.txt', 'w')

	problem =1

	for problem in range(3, 4): 
 

		if problem == 0:

			 
			dataset = [[2.7810836,2.550537003,0],
				[1.465489372,2.362125076,0],
				[3.396561688,4.400293529,0],
				[1.38807019,1.850220317,0],
				[3.06407232,3.005305973,0],
				[7.627531214,2.759262235,1],
				[5.332441248,2.088626775,1],
				[6.922596716,1.77106367,1],
				[8.675418651,-0.242068655,1],
				[7.673756466,3.508563011,1]]


			traindata = np.asarray(dataset) # convert list data to numpy
			testdata = train_data
			classi_reg = True  # true for classification and false for regression
			features = 2  #
			output = 1

	  

		if problem == 1:
			traindata = np.loadtxt("data/Lazer/train.txt")
			testdata = np.loadtxt("data/Lazer/test.txt")  

			classi_reg = False  # true for classification and false for regression
			features = 4  #
			output = 1
#
		if problem == 2:
			traindata = np.loadtxt("data/Sunspot/train.txt")
			testdata = np.loadtxt("data/Sunspot/test.txt")  #

			classi_reg = False  # true for classification and false for regression
			features = 4  #
			output = 1

		if problem == 3:
			traindata = np.loadtxt("data/Mackey/train.txt")
			testdata = np.loadtxt("data/Mackey/test.txt")  # 
			classi_reg = False  # true for classification and false for regression
			features = 4  #
			output = 1


		print(traindata)

		topology = [features, output]

		MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)


		numSamples = 10000  # need to decide yourself

		mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

		[pos_w, pos_tau, fx_train, fx_test,   rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
		print('sucessfully sampled')

		burnin = 0.25 * numSamples  # use post burn in samples

		pos_w = pos_w[int(burnin):, ]
		pos_tau = pos_tau[int(burnin):, ]

		fx_mu = fx_test.mean(axis=0)
		fx_high = np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr = np.percentile(fx_train, 5, axis=0)

		rmse_tr = np.mean(rmse_train[int(burnin):])
		rmsetr_std = np.std(rmse_train[int(burnin):])
		rmse_tes = np.mean(rmse_test[int(burnin):])
		rmsetest_std = np.std(rmse_test[int(burnin):])
		print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
		np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

		ytestdata = testdata[:, input]
		ytraindata = traindata[:, input]

		if classi_reg == False:

			plt.plot(x_test, ytestdata, label='actual')
			plt.plot(x_test, fx_mu, label='pred. (mean)')
			plt.plot(x_test, fx_low, label='pred.(5th percen.)')
			plt.plot(x_test, fx_high, label='pred.(95th percen.)')
			plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
			plt.legend(loc='upper right')

			plt.title("Plot of Test Data vs MCMC Uncertainty ")
			plt.savefig('mcmcresults/mcmcrestest.png')
			plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
			plt.clf()
			# -----------------------------------------
			plt.plot(x_train, ytraindata, label='actual')
			plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
			plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
			plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
			plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
			plt.legend(loc='upper right')

			plt.title("Plot of Train Data vs MCMC Uncertainty ")
			plt.savefig('mcmcresults/mcmcrestrain.png')
			plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
			plt.clf()

		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)

		ax.boxplot(pos_w)

		ax.set_xlabel('[W1] [B1] [W2] [B2]')
		ax.set_ylabel('Posterior')

		plt.legend(loc='upper right')

		plt.title("Boxplot of Posterior W (weights and biases)")
		plt.savefig('mcmcresults/w_pos.png')
		plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)

		plt.clf()


if __name__ == "__main__": main()


 