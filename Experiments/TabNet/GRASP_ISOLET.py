from Algorithms.GRASP import AdaptedGRASPForTabnet
from time import time
import numpy as np
from Datasets.ISOLET import dataset
from Models import tabnet

# Parameters used in the paper
iter = 10 # Every selected solution is iterated by iter = 10
alfa = 0.20 # alfa - A parameter determines how greedy or random the List Restrict Candidates will be.
Pmax = 20 # In each iteration Pmax = 20
p = 0.5 # Probability of changing neurons at the particular layer
K = 0.03 # The updation of neurons (K) is set to 3%
maxim = 5 # Maximum amount of hidden layers

inicio = time()
Xtrain, X_test, ytrain, y_test = dataset(return_tensor=False)
best_isolet, lista_melhores = AdaptedGRASPForTabnet.grasp(Xtrain, ytrain, max=5, iter=10)

for i in best_isolet:
    acc, val, test = tabnet.test_model(i, Xtrain, X_test, ytrain, y_test)
    print("\nBest results for ISOLET:")
    print('N_d:', i.N_d)
    print('N_a:', i.N_a)
    print('L_r:', i.l_r)
    print('N_steps:', i.N_steps)
    print('error_valid:', i.error_valid)
    print('std_valid:', i.std_valid)
    print("Testing:")
    print('Train Accuracy: {:.6f}'.format(np.mean(acc)))
    print('Standard Deviation Train Accuracy: {:.6f}'.format(np.std(acc)))
    print('Average Error Valid: {:.6f}'.format(np.mean(val)))
    print('Standard Deviation Error Valid: {:.6f}'.format(np.std(val)))
    print('Error Test: {:.6f}'.format(np.mean(test)))
    print('Standard Deviation Error Test: {:.6f}'.format(np.std(test)))

print("\nTotal model time: {:.5f} minutes".format((time()-inicio) /60))
print(lista_melhores)
print()
