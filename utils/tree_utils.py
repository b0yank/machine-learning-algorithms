import numpy as np
from abc import abstractmethod

MAX_FEATURES = {
    'auto': lambda x: np.ceil(np.sqrt(x)).astype('int'),
    'sqrt': lambda x: np.ceil(np.sqrt(x)).astype('int'),
    'log2': lambda x: np.ceil(np.log2(x)).astype('int'),
    None: lambda x: x
}
CRITERIA_CLF = {
    'gini': lambda p: 1 - np.sum(p ** 2), 
    'entropy': lambda p: -np.sum(p * np.log2(p))
}

STARTING_SCORE = float('-inf')

class TreeClassifier:
    @abstractmethod
    def __init__(self, 
                 criterion,
                 max_depth, 
                 min_samples_split, 
                 min_samples_leaf,
                 max_features,
                 random_state):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self._criterion = CRITERIA_CLF[criterion]
        self._max_depth = max_depth if max_depth != None else sys.maxsize
        
        self._rand_gen = np.random.RandomState()
        if isinstance(random_state, int):
            self._rand_gen.seed(random_state)
        elif isinstance(random_state, type(np.random.RandomState())):
            self._rand_gen = random_state
            
    @abstractmethod
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if isinstance(self.max_features, (str, type(None))):
            self._n_features = MAX_FEATURES[self.max_features](n_features)
        elif isinstance(self.max_features, float) and self.max_features > 0 and self.max_features <= 1.0:
            self._n_features = np.ceil(self.max_features * n_features).astype('int')
        elif isinstance(self.max_features, int):
            self._n_features = self.max_features if self.max_features < n_features else n_features
        else:
            raise ValueError(f'Invalid value for max_features: {self.max_features}')
            
        if isinstance(self.min_samples_leaf, float):
            self.min_leaf = np.ceil(n_samples * self.min_samples_leaf)
        elif isinstance(self.min_samples_leaf, int):
            self.min_leaf = self.min_samples_leaf
        else:
            raise ValueError('Parameter min_samples_leaf must be either an integer or a floating point number.')
            
        if isinstance(self.min_samples_split, float):
            self.min_split = np.ceil(n_samples * self.min_samples_split)
        elif isinstance(self.min_samples_split, int):
            self.min_split = self.min_samples_split
        else:
            raise ValueError('Parameter min_samples_split must be either an integer or a floating point number.')




