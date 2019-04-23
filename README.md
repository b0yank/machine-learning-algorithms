# machine-learning-algorithms
Contains implementations of the core machine learning algorithms.

The implementations of the following classic ML algorithms imitate scikit-learn's API:
- **LinearRegression**
- **LogisticRegression**
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **SVC (support vector classifier)**
- **Perceptron**

Currently, the repository also contains implementations of several types of neural network layers and the Sequential neural network model, all inspired by the Keras API:
- **Activation layer**
- **BatchNormalization layer**
- **Conv2D layer**
- **Dense layer**
- **Dropout layer**
- **Flatten layer**
- **Embedding layer**
- **Masking layer**
- **SimpleRNN layer**
- **GRU layer**
- **LSTM layer**
- **Sequential model**

The repository currently supports stochastic gradient descent (**SGD**) and **Adam** optimizers. In terms of loss functions, the choice is between **cross-entropy** and **mean squared loss**.

Please note that I am still learning and am not a professional practitioner. I am in the process of learning the details of ML/DL algorithms and architectures, and it is very likely that some of the implementations may **not** be entirely correct or bug-free.

In addition, please bear in mind that these implementations use numpy and are not computationally optimized. I also, admittedly, waste a lot of RAM at certain places (e.g. the GRU and LSTM layers) because the purpose of the library is educational and making the equations clear is more important to me than striving for practical usability (although I have tested them on simple tasks).
