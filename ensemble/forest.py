import numpy as np

from ..tree import DecisionTreeClassifier
from ..utils import TreeClassifier

class RandomForestClassifier(TreeClassifier):
    def __init__(self,
                n_estimators = 10,
                criterion = 'gini',
                max_depth = None,
                min_samples_split = 2,
                min_samples_leaf = 1,
                max_features = 'auto',
                bootstrap = True,
                oob_score = False,
                random_state = None):
        super().__init__(
                criterion = criterion,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                random_state = random_state)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
            
        self.__trees = [DecisionTreeClassifier(criterion = criterion,
                                               max_depth = max_depth,
                                               min_samples_split = min_samples_split,
                                               min_samples_leaf = min_samples_leaf, 
                                               random_state = random_state) for t in range(n_estimators)]
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.oob_score:
            if not self.bootstrap:
                raise ValueError('Out of bag estimation only available if bootstrap=True')
                
            oob_idxs = {}
        
        X_idxs = range(n_samples)
            
        for tree in self.__trees:
            # bootstrap samples for current tree
            idxs = self._rand_gen.choice(X_idxs, n_samples, replace = self.bootstrap)
            tree.fit(X[idxs], y[idxs])
            
            # keep track of the out-of-bag samples
            if self.oob_score:
                oob_idxs[tree] = {idx for idx in set(X_idxs) if idx not in idxs}
       
        if self.oob_score:
            self.__compute_oob(X, y, oob_idxs)
     
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.__trees]).T
        
        # iterate predictions for each sample and get the mode
        preds_final = [mode(prediction)[0][0] for prediction in predictions]
        return preds_final
            
    def score(self, X, y):
        n_samples = len(y)
        pred = self.predict(X)
        return (n_samples - np.count_nonzero(y - pred)) / n_samples
    
    def __compute_oob(self, X, y, oob_idxs):
        n_samples = X.shape[0]
        sample_range = range(n_samples)
        oob_scores = {key: [] for key in sample_range}
        oob_samples = set()
        for index in sample_range:
            for tree in self.__trees:
                if index in oob_idxs[tree]:
                    oob_scores[index].append(tree.predict(X[index].reshape(1, -1))[0])
                    oob_samples.add(index)
        
        if len(oob_samples) < n_samples:
            warn('Some inputs do not have OOB scores. '
                     'This probably means too few trees were used '
                    'to compute any reliable oob estimates.')
                        
        oob_predictions = [mode(oob_scores[key])[0][0] for key in oob_samples]
        self.oob_score_ = (len(oob_samples) - np.count_nonzero(y[list(oob_samples)] - oob_predictions)) / len(oob_samples)

