#import pandas as pd
import numpy as np

# FUNCTIONS

# Normalization
def normalize(data, data_val):
    outdata = data.copy()
    outdata_val = data_val.copy()
    for it in range(data.shape[1]):
        mean = np.mean(data[:,it])
        std = np.std(data[:,it])
        outdata[:,it] = (outdata[:,it] - mean) / std
        outdata_val[:,it] = (outdata_val[:,it] - mean) / std
    return outdata, outdata_val

class nNetwork:
    def __init__(self, nhidden, eta, mini_batch_size):
        self.nhidden = nhidden
        self.eta = eta
        self.p = mini_batch_size


    # Initialization of weights and thresholds
    def _init_weights_W(self):
        return np.random.normal(loc = 0, scale =np.sqrt(1/self.nhidden), size =
                               self.nhidden)
    def _init_weights_w(self):
        return np.random.normal(loc = 0, scale = np.sqrt(0.5), size =
                                (self.nhidden,self.n))
    def _init_thresholds(self):
        return np.zeros(self.nhidden), np.zeros(self.O)


    # Local fields 
    def _calc_local_field_b_mat(self, x_mat, w_mat, theta_vec):
        p, n = x_mat.shape
        b_mat = np.zeros((self.nhidden, p))
        for mu in range(self.p):
            for jt in range(self.nhidden):
                b_mat[jt, mu] = -theta_vec[jt]
                for kt in range(n):
                    b_mat[jt, mu] += w_mat[jt,kt]*x_mat[mu,kt]
        return b_mat
    def _calc_local_field_B(self, V_mat, W_mat, theta_vec):
        p = V_mat.shape[1]
        B_mat = np.zeros((self.O, p))
        for mu in range(p):
            for it in range(self.O):
                tmp_it = 0
                for jt in range(self.nhidden):
                    tmp_it += W_mat[it, jt] * V_mat[jt,mu]
                B_mat[mu,it] = tmp_it - theta_vec[it]
        return B_mat


    # Neuron calucations
    def _calc_neurons(self, b_mat): # implement ReLU
        return np.tanh(b_mat)
    #def _calc_output(self, B_vec):
    #    return np.tanh(B_vec)

    # Softmax function
    def _softmax(self, B_mat):
        p = B_mat.shape[1]
        O_mat = np.zeros((p, self.O))
        for mu in range(p):
            tmp_max = np.max(B_mat[:,p])
            y = np.exp(B_mat[:,mu] - tmp_max)
            f_b = y / np.sum(np.exp(B_mat[:,mu]))
            O_mat[mu,:] = f_b
        return O_mat


    # Back propagation step 1
    def _deriv_tanh(self,bi):
        return 1 - np.tanh(bi)**2
    def _calc_Delta(self, t_vec,O_vec, B_vec):
        Delta_vec = np.zeros(self.p)
        for mu in range(self.p):
            Delta_vec[mu] =  (t_vec[mu] - O_vec[mu]) #* self._deriv_tanh(B_vec[mu])
        return Delta_vec
    def _calc_change_W_vec(self, Delta_vec, V_mat):
        return self.eta * np.matmul(V_mat, Delta_vec)
    def _calc_change_Theta(self, Delta_vec):
        return -self.eta * sum(Delta_vec)

    # Back propagation step 2
    def _calc_delta_mat(self, Delta_vec, W_mat, b_mat):
        delta_mat = np.zeros((self.nhidden,self.p))
        for mu in range(self.p):
            for jt in range(self.nhidden):
                delta_mat[jt,mu] = Delta_vec[mu] * W_mat[jt] * self._deriv_tanh(b_mat[jt,mu])
        return delta_mat
    def _calc_change_w_mat(self, delta_mat, x_mat):
        return self.eta * np.matmul(delta_mat, x_mat)
    def _calc_change_theta_vec(self, delta_mat):
        return -self.eta * np.sum(delta_mat, axis = 1)

    # Update weights
    def _update_weights(self, W_mat, w_mat, change_W, change_w):
        return np.add(W_mat, change_W), np.add(w_mat, change_w)
    def _update_thresholds(self, theta_vec, Theta, change_theta_vec, change_Theta):
        return np.add(theta_vec, change_theta_vec), np.add(Theta, change_Theta)

    # Classification error
    def _calc_C(self, t_vec, O_vec):
        tmp = np.sum( np.absolute( np.subtract( np.sign(O_vec), t_vec ) ) ) /(2*t_vec.shape[0])
        return tmp
    def _class_error_validation(self, x_val, t_val, w_mat, W_mat, theta_vec, Theta):
            b_mat = self._calc_local_field_b_mat(x_val, w_mat, theta_vec)
            V_mat = self._calc_neurons(b_mat)
            B_vec = self._calc_local_field_B(V_mat, W_mat, Theta)
            O_vec = self._calc_output(B_vec)
            return self._calc_C(t_val, O_vec), O_vec


    # Running the network
    def _run_training(self, x_mat, t_vec, x_val, t_val):
        # Initialize
        W_mat = self._init_weights_W()
        w_mat = self._init_weights_w()
        theta_vec, Theta_vec = self._init_thresholds()
        val_C = np.zeros(self.epochs)
        best_val_C = np.inf
        # Training loop  
        for ep in range(self.epochs):
            for it in range(x_mat.shape[0]):
                # Select minibatch
                indices = np.random.choice(range(x_mat.shape[0]), size = self.p)
                x_it = np.copy(x_mat[indices,:])
                t_it = np.copy(t_vec[indices])
                # Forward feed
                b_mat = self._calc_local_field_b_mat(x_it, w_mat, theta_vec)
                V_mat = self._calc_neurons(b_mat)
                B_vec = self._calc_local_field_B(V_mat, W_mat, Theta_vec)
                O_vec = self._softmax(B_vec)
                # Back prop
                Delta_vec = self._calc_Delta(t_it, O_vec, B_vec)
                change_W_vec = self._calc_change_W_vec(Delta_vec, V_mat)
                change_Theta = self._calc_change_Theta(Delta_vec)
                delta_mat = self._calc_delta_mat(Delta_vec, W_mat, b_mat)
                change_w_mat = self._calc_change_w_mat(delta_mat, x_it)
                change_theta_vec = self._calc_change_theta_vec(delta_mat)
                # Update weights
                W_mat, w_mat = self._update_weights(W_mat, w_mat, change_W_vec, change_w_mat)
                theta_vec, Theta_vec = self._update_thresholds(theta_vec, Theta, change_theta_vec, change_Theta)

            # Classification error validation set
            val_C[ep], output = self._class_error_validation(x_val,t_val, w_mat,
                                                             W_mat, theta_vec,
                                                             Theta_vec)
            print("Validation error of epoch ", ep, ": ", val_C[ep])
            if (val_C[ep] < best_val_C):
                best_val_C = val_C[ep]
                best_w = w_mat
                best_W = W_mat
                best_t = theta_vec
                best_T = Theta_vec
        return best_w, best_W, best_t, best_T

    def train(self, x_mat, t_vec, x_val, t_val, epochs):
        self.O = 10
        self.epochs = epochs
        self.n = x_mat.shape[1]
        self.w1, self.w2, self.theta1, self.theta2 = self._run_training(x_mat, t_vec,x_val, t_val)
        print("Training done")

    def predict(self, x_mat, t_vec):
        C, output_vec = self._class_error_validation(x_mat, t_vec, self.w1, self.w2, self.theta1, self.theta2)
        return output_vec, C


### Calculations ###

# Load data
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape((train_X.shape[0],28*28))
test_X = test_X.reshape((test_X.shape[0], 28*28))

# Run training
nnet = nNetwork(50, 0.005, 10)
#nnet.train(x_mat,t_vec,x_val,t_val, 10)


# Load data
#colnames = ['x1', 'x2', 't']
#train = pd.read_csv('training_set.csv', names = colnames)
#validation = pd.read_csv('validation_set.csv', names = colnames)
#train_norm, validation_norm = normalize(train.to_numpy(), validation.to_numpy())
#x_mat = train_norm[:,:2]
#t_vec = train_norm[:,2]
#x_val = validation_norm[:,:2]
#t_val = validation_norm[:,2]
