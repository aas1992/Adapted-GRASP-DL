from Algorithms.GRASP import AdaptedGRASPForDFNN
from time import time
import numpy as np
from Datasets.ISOLET import dataset

# Parameters used in the paper
iter = 10 # Every selected solution is iterated by iter = 10
alfa = 0.20 # alfa - A parameter determines how greedy or random the List Restrict Candidates will be.
Pmax = 20 # In each iteration Pmax = 20
p = 0.5 # Probability of changing neurons at the particular layer
K = 0.03 # The updation of neurons (K) is set to 3%
maxim = 5 # Maximum amount of hidden layers

inicio = time()
dataset_train, dataset_test = dataset(return_tensor=True)
best_isolet, lista_melhores = AdaptedGRASPForDFNN.grasp(617, 26, dataset_train, maxim=5, iter=10)
print("\nBest results for ISOLET:")
for i in best_isolet:
  print("Calculating the fitness of HL={} and HN={}".format(i.HL, i.N_List))
  print('Error Valid: {:.6f}'.format(i.error_valid))
  print('Standard deviation Valid: {:.6f}'.format(i.std_valid))
  print("Testing...")
  train_acc, test_acc = AdaptedGRASPForDFNN.test_model(i, 617, 26, dataset_train, dataset_test)
  print('Average Acc Train:', np.mean(train_acc))
  print('Standard deviation Train:', np.std(train_acc))
  print('Average Acc Test:', np.mean(test_acc))
  print('Standard deviation Test:', np.std(test_acc))
print("\nTotal model time: {} minutes".format((time()-inicio)/60))

# Save the best information for plotting charts
print(lista_melhores)
print()