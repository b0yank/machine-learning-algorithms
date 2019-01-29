import cvxopt
import numpy as np
import sys, os

class ModelNotTrainedException(Exception):
    pass

SVM_KERNELS = {
    'linear': lambda x_0, x_1, gamma, degree, coef0: np.inner(x_0, x_1),
    'rbf': lambda x_0, x_1, gamma, degree, coef0: np.exp((np.linalg.norm(x_0 - x_1) ** 2) * -gamma),
    'poly': lambda x_0, x_1, gamma, degree, coef0: (gamma *  np.inner(x_0, x_1) + coef0) ** degree,
    'sigmoid': lambda x_0, x_1, gamma, degree, coef0: np.tanh(gamma * np.inner(x_0, x_1) + coef_0)
}

class SupportVectorClassifier:
    """
    A binary support vector classifier.
    """
    def __init__(self, class_, kernel, C, degree,  gamma, coef0):
        self.class_ = class_
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
    def fit(self, X, y):
        self.n_samples = len(X)
        self.K = self.gram_matrix(X)
        lagrange_multipliers = self.get_lagrange_multipliers(X, y)
        
        MIN_SV_MULTIPLIER = 1e-5
        # only consider multipliers larger than MIN_SV_MULTIPLIER as support vectors
        sv_indices = lagrange_multipliers > MIN_SV_MULTIPLIER
        self.multipliers = lagrange_multipliers[sv_indices]
        self.supp_vec = X[sv_indices]
        self.sv_labels = y[sv_indices]
        
        inside_indices = (lagrange_multipliers > MIN_SV_MULTIPLIER) & (lagrange_multipliers < self.C)
        inside_vectors = X[inside_indices]
        inside_labels = y[inside_indices]
        
        # reform Gram matrix to only contain support vectors
        self.sv_count = len(self.supp_vec)
        self.K = self.K[np.outer(sv_indices, sv_indices)]
        self.K.shape = (self.sv_count, self.sv_count)
        
        M = len(inside_vectors)
        
        # bias = (1/M) * sum(y_n - sum(a_m * y_m * K[n, m]))
        bias = \
            (1/M) * np.sum([y[n] - np.sum([self.multipliers[m] * self.sv_labels[m] * self.K[n, m]\
                                           for m in range(self.sv_count)])\
                                                for n in range(M)])
        
        self.bias = bias
        
    def predict(self, X):
        y_x = np.array([np.sum([
                self.multipliers[n] *\
                    self.sv_labels[n] * self.kernel(x, self.supp_vec[n], self.gamma, self.degree, self.coef0)\
                        for n in range(self.sv_count)]) for x in X]) + self.bias
        
        return y_x, np.sign(y_x).astype('int64')
        
        
    def gram_matrix(self, X):
        K = np.zeros((self.n_samples, self.n_samples))
        for n, x_n in enumerate(X):
            for m, x_m in enumerate(X):
                K[n, m] = self.kernel(x_n, x_m, self.gamma, self.degree, self.coef0)
        return K

    def get_lagrange_multipliers(self, X, y):
        """
        Maximizes the Lagrangian function:
            L = sum(a_n) - 0.5 * sum_n(sum_m(a_n * a_m * y_n * y_m * K[n, m]))  // a_n, a_m - Lagrange multipliers
                                                                                // K - gram matrix
                                                                                
        Subject to constraints:
            0 <= a_n <= C  // C - regularization parameter
            sum_n(a_n * y_n) = 0
            
        Note: This method actually minimizes -L, which is equivalent to maximizing L
        """                 
        # y_n * y_m * K[n, m]
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        # negative multiplier for sum(a_n)
        q = cvxopt.matrix(-1 * np.ones(self.n_samples))
        
        # constraint -a_n <= 0    
        G_std = cvxopt.matrix(np.diag(np.ones(self.n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(self.n_samples))
        
        # constraint a_n <= C
        G_slack = cvxopt.matrix(np.diag(np.ones(self.n_samples)))
        h_slack = cvxopt.matrix(np.ones(self.n_samples) * self.C)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        
        # constraint sum(a_n * y_n) = 0 // A - matrix of y-values
        #                               // b - matrix of zeros representing right-hand side
        A = cvxopt.matrix(y, (1, self.n_samples), 'd')
        b = cvxopt.matrix(0.0)

        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # minimize Lagrangian function
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        sys.stdout = stdout

        # return Lagrange multipliers
        return np.ravel(solution['x'])

class SVC:
    """
    Adapts multiple, traditional binary support vector classifiers for multiclassification purposes.
    """
    def __init__(self, kernel = 'rbf', C = 1.0, degree = 3, gamma = 'auto', coef0 = 0.0):
        if C < 0:
            raise ValueError('C must be a non-negative number.')
            
        GAMMA_ERROR = 'Gamma value of {} invalid. Use \'auto\' to set gamma to 1 / n_features'
        if gamma == 0:
            raise ValueError(GAMMA_ERROR.format(gamma))
        if (type(gamma) is str) & (gamma != 'auto'):
            raise ValueError(GAMMA_ERROR.format(gamma))
        
        if kernel not in SVM_KERNELS.keys():
            raise ValueError('Invalid kernel - choose between \'linear\', \'rbf\', \'poly\' and \'sigmoid\'')
        
        self.gamma = gamma
        self.__gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.__is_trained = False
        self.__kernel = SVM_KERNELS[kernel]

    def fit(self, X, y):
        if self.gamma == 'auto':
            self.__gamma = 1 / X.shape[1]
        
        self.__classes = list(set(y))
        n_samples = len(X)
        n_classes = len(self.__classes)
        train_classes = [self.__classes[0]] if n_classes == 2 else self.__classes
        
        self.__classifiers = []
        for class_ in train_classes:
            positive_idxs = np.argwhere(y == class_)
            negative_idxs = np.setdiff1d(np.arange(n_samples), positive_idxs, assume_unique = True)
            y_train = y.copy()
            y_train[positive_idxs] = 1
            y_train[negative_idxs] = -1
            
            classifier = SupportVectorClassifier(class_, self.__kernel, self.C, self.degree, self.__gamma, self.coef0)
            
            classifier.fit(X, y_train)
            self.__classifiers.append(classifier)
        
        self.__is_trained = True
        
    def predict(self, X):
        if self.__is_trained == False:
            raise ModelNotTrainedException('Model needs to be trained first.')
            
        n_samples = len(X)
        
        # initial results set to second class 
        # to automatically take care of negative class in case of binary classification
        results_preds = [(float('-inf'), self.__classes[1]) for i in range(n_samples)]
        
        for classifier in self.__classifiers:
            results, predictions = classifier.predict(X)
            positive_idxs = np.argwhere(predictions == 1)
            
            for index in positive_idxs:
                # class corresponding to classifier with the largest y(x) value will be predicted
                # this is to avoid a case where multiple classes are assigned
                if results[index] > results_preds[int(index)][0]:
                    results_preds[int(index)] = (results[index], classifier.class_)
            
        return [res_pred[1] for res_pred in results_preds]
    
    def score(self, X, y):
        if self.__is_trained == False:
            raise ModelNotTrainedException('Model needs to be trained first.')
        
        predictions = self.predict(X)
        diff= y - predictions
        return 1 - (np.count_nonzero(diff) / len(y))