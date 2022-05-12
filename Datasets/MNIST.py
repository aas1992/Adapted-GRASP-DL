import torch # for Deep Learning
import numpy as np
import torch.utils.data as data_utils
from keras.datasets import mnist     # MNIST dataset is included in Keras

# dir_remoto = "/content/drive/My Drive/"
# dir_local = os.getcwd() # path para rodar em máquina local ao invés do colab

# Separate Data into Training, Validation and Testing
def dataset(return_tensor=True):
    (xtrain, ytrain), (xtest, y_test) = mnist.load_data()

    x_train = [item.reshape(1,28*28) for item in xtrain]
    X_train = [x_train[n][0] for n in range(len(x_train))]
    x_test = [item.reshape(1,28*28) for item in xtest]
    X_test = [x_test[n][0] for n in range(len(x_test))]

    Xtrain = np.array(X_train)
    print(Xtrain)
    print(Xtrain.shape)
    X_test = np.array(X_test)
    print(X_test.shape)

    dataset = data_utils.TensorDataset(torch.FloatTensor(Xtrain), torch.LongTensor(ytrain)) # Convertendo para Tensor
    dataset_test = data_utils.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)) # Convertendo para Tensor

    if return_tensor:
        return dataset, dataset_test
    else:
        return Xtrain, X_test, ytrain, y_test