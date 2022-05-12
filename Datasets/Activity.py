import torch # for Deep Learning
import pandas as pd
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using PyTorch version:', torch.__version__, ' Device:', device)

# dir_remoto = "/content/drive/My Drive/"
# dir_local = os.getcwd() # path para rodar em máquina local ao invés do colab
def dataset(return_tensor=True):
    data = pd.read_csv('D:/Andersson.com/Documentos/Doutorado - CIn/2º SEM/DEEP LEARNING/Projeto de DL/IJCNN 2022/Adapted-GRASP-DL-main/Datasets/Activity_Recognition_Using-WPM.txt')

    print("Data")
    print(data)
    print(data.shape)

    # Convert to numpy array
    data = data.to_numpy()

    # Separate Data into Training, Validation and Testing
    xtrain = data[:, 1:534]
    ytrain = data[:, 534]
    ytrain = ytrain.astype('int')
    ytrain = ytrain-1

    xtrain = minmax_scale(xtrain)

    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=.10, random_state=1)

    print("TRAIN")
    print(X_train.shape)
    #print(X_train)
    print(y_train.shape)
    #print(y_train)

    print("\nTEST")
    print(X_test.shape)
    #print(X_test)
    print(y_test.shape)
    #print(y_test)

    dataset = data_utils.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)) # Converting to Tensor
    dataset_test = data_utils.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))  # Converting to Tensor

    if return_tensor:
        return dataset, dataset_test
    else:
        return X_train, X_test, y_train, y_test