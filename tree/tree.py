import numpy as np

from ..utils import TreeClassifier

class Node:
    def __init__(self, X, y, depth, criterion, min_samples_split, min_samples_leaf, max_depth):
        self.y = y
        self.depth = depth
        
        if self.is_pure or depth == max_depth:
            self.setup_leaf()
            return
        
        self.X = X
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.leaf = False
        self.score = STARTING_SCORE
        self.split_value = None
        self.col_index = -1
        
        self.split()
    
    @property
    def is_leaf(self): return self.leaf
    
    @property
    def is_pure(self): return len(set(self.y)) == 1
    
    def split(self):
        if self.min_samples_split > len(self.X):
            self.setup_leaf()
            return
            
        best_left = []
        best_right = []
        feature_idxs = range(self.X.shape[1])
        for col_idx in feature_idxs:
            left_ind, right_ind = self.get_best_split(col_idx)
            
            # check if best score was updated
            if len(left_ind) > 0:
                best_left = left_ind
                best_right = right_ind
        
        # if score wasn't updated, we are at a leaf
        if self.score == STARTING_SCORE:
            self.setup_leaf()
            return
               
        self.left_child = Node(self.X[best_left],
                               self.y[best_left],
                               self.depth + 1,
                               self.criterion,
                               self.min_samples_split,
                               self.min_samples_leaf,
                               self.max_depth)
        self.right_child = Node(self.X[best_right],
                                self.y[best_right],
                                self.depth + 1,
                                self.criterion,
                                self.min_samples_split,
                                self.min_samples_leaf,
                                self.max_depth)
    
    def get_information_gain(self, left_ind, right_ind):
        left_y = self.y[left_ind]
        right_y = self.y[right_ind]
        parent_y = np.concatenate((left_y, right_y))
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        
        left_info = self.criterion(np.unique(left_y, return_counts = True)[1] / n_left)
        right_info = self.criterion(np.unique(right_y, return_counts = True)[1] / n_right)
        parent_info = self.criterion(np.unique(parent_y, return_counts = True)[1] / n_total)
        
        gain = parent_info - (n_left / n_total) * left_info - (n_right / n_total) * right_info
        
        return gain
    
    def split_column(self, column, value):
        left_ind = np.nonzero(column >= value)[0]
        right_ind = np.nonzero(column < value)[0]
        
        return left_ind, right_ind
    
    def setup_leaf(self):
        self.leaf = True
        values, counts = np.unique(self.y, return_counts = True)
        self.label = values[np.argmax(counts)]
    
    def get_best_split(self, col_index):
        MAX_VALS = 30
        column = self.X[:, col_index]
        unique_vals = sorted(np.unique(column))
        val_count = len(unique_vals)
        best_left, best_right = [], []
        
        # iterate through not more than MAX_VALS unique values
        step = 1 if val_count <= MAX_VALS else int(round(val_count / MAX_VALS))
        
        for index in range(1, val_count, step):
            value = unique_vals[index]
            left_ind, right_ind = self.split_column(column, value)
            if len(left_ind) < self.min_samples_leaf or len(right_ind) < self.min_samples_leaf:
                continue
                
            current_score = self.get_information_gain(left_ind, right_ind)
            
            if current_score > self.score:
                self.score = current_score
                self.split_value = value
                self.col_index = col_index
                best_left, best_right = left_ind, right_ind
                
        return best_left, best_right
    
    def predict(self, x_i):
        if self.leaf:
            return self.label
        
        compared_value = x_i if np.isscalar(x_i) else x_i[self.col_index]
        if compared_value >= self.split_value:
            return self.left_child.predict(x_i)

        return self.right_child.predict(x_i)

class DecisionTreeClassifier(TreeClassifier):
    def __init__(self, 
                 criterion = 'gini',
                 max_depth = None, 
                 min_samples_split = 2, 
                 min_samples_leaf = 1,
                 max_features = None,
                 random_state = None): 
        super().__init__(
                criterion = criterion,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                random_state = random_state)
        
    def fit(self, X, y):
        super().fit(X, y)
        n_samples, n_features = X.shape
        
        feature_idxs = []
        if n_features > self._n_features:
            feature_idxs = self._rand_gen.choice(range(n_features), self.max_features, replace = False)
        else:
            feature_idxs = range(n_features)
        
        self.__root = Node(X,
                         y,
                         depth = 1,
                         criterion = self._criterion,
                         min_samples_split = self.min_split,
                         min_samples_leaf = self.min_leaf,
                         max_depth = self._max_depth)
        
    def predict(self, X): return [self.__root.predict(x_i) for x_i in X]
    
    def score(self, X, y):
        predictions = self.predict(X)
        n_samples = len(y)
        
        return (n_samples - np.count_nonzero(y - predictions)) / n_samples

