 
 # by R. Chandra
 #Source: https://github.com/rohitash-chandra/logistic_regression

from math import exp
import numpy as np
import random

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

		print(self.w, ' self.w init') 
		print(self.b, ' self.b init') 
		print(self.out_delta, ' outdel init')


 
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
	
	def gradient(self, x_vec, output, actual):   
		if self.use_activation == SIGMOID :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =   (output - actual) 
		return out_delta

	def update(self, x_vec, output, actual):      
		self.w+= self.learn_rate *( x_vec *  self.out_delta)
		self.b+=  (1 * self.learn_rate * self.out_delta)
 

	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class
 


	def test_model(self, data, tolerance):  

		num_instances = data.shape[0]

		class_perf = 0
		sum_sqer = 0   
		for s in range(0, num_instances):	

			input_instance  =  self.train_data[s,0:self.num_features] 
			actual  = self.train_data[s,self.num_features:]  
			prediction = self.predict(input_instance) 
			sum_sqer += self.squared_error(prediction, actual)

			pred_binary = np.where(prediction > (1 - tolerance), 1, 0)

			print(s, actual, prediction, pred_binary, sum_sqer, ' s, actual, prediction, sum_sqer')

 

			if( np.sum(actual-pred_binary)==0):
				class_perf =  class_perf +1  

		rmse = np.sqrt(sum_sqer/num_instances)

		percentage_correct = float(class_perf/num_instances) * 100 

		print(percentage_correct, rmse,  ' class_perf, rmse') 
		# note RMSE is not a good measure for multi-class probs

		return ( rmse, percentage_correct)





 
	def SGD(self):   
		
			epoch = 0 
			shuffle = True

			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 

					if shuffle ==True:
						i = random.randint(0, self.num_train-1)

					input_instance  =  self.train_data[i,0:self.num_features]  
					actual  = self.train_data[i,self.num_features:]  
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual)
					self.out_delta = self.gradient( input_instance, prediction, actual)    # major difference when compared to GD
					#print(input_instance, prediction, actual, s, sum_sqer)
					self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.train_data, 0.3) 
			rmse_test =0
			test_perc =0
			#rmse_test, test_perc = self.test_model(self.test_data, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				

	def GD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.train_data[s,0:self.num_features]  
					actual  = self.train_data[s,self.num_features:]   
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual) 
					self.out_delta+= self.gradient( input_instance, prediction, actual)    # this is major difference when compared with SGD

					#print(input_instance, prediction, actual, s, sum_sqer)
				self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.train_data, 0.3) 
			rmse_test =0
			test_perc =0
			#rmse_test, test_perc = self.test_model(self.test_data, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				
	
 







#------------------------------------------------------------------


class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        random.seed()
	 

        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
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

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print y_train.size
        print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.02;  # defines how much variation you need in changes to w
        step_eta = 0.01;
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)
        print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print likelihood

        naccept = 0
        print 'begin sampling using mcmc random walk'
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print    i, ' is accepted sample'
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro

                print  likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted'

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

        print naccept, ' num accepted'
        print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)











def main():
  

    outres = open('results.txt', 'w')

    problem =1

    for problem in xrange(2, 3): 
 

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

        random.seed(time.time())

        numSamples = 80000  # need to decide yourself

        mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print 'sucessfully sampled'

        burnin = 0.1 * numSamples  # use post burn in samples

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
        print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
        np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        ytestdata = testdata[:, input]
        ytraindata = traindata[:, input]

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





























'''def main(): 

	random.seed()
	 

	 
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


	train_data = np.asarray(dataset) # convert list data to numpy
	test_data = train_data

	 

	learn_rate = 0.3
	num_features = 2
	num_epocs = 20

	print(train_data)
	 

	lreg = logistic_regression(num_epocs, train_data, test_data, num_features, learn_rate)
	(train_perc, test_perc, rmse_train, rmse_test) = lreg.SGD()
	(train_perc, test_perc, rmse_train, rmse_test) = lreg.GD() 
	 

	#-------------------------------
	#xor data


	xor_dataset= [[0,0,0],
		[0,1,1],
		[1,0,1],
		[1,1,0] ]

	xor_data = np.asarray(xor_dataset) # convert list data to numpy



	num_epocs = 20
	learn_rate = 0.9
	num_features = 2

	lreg = logistic_regression(num_epocs, xor_data, xor_data, num_features, learn_rate)
	(train_perc, test_perc, rmse_train, rmse_test) = lreg.SGD()
	(train_perc, test_perc, rmse_train, rmse_test) = lreg.GD() 


if __name__ == "__main__": main()'''