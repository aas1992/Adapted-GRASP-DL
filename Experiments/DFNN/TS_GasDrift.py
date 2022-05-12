from Algorithms.TabuSearch import TabuSearchForDFNN
from time import time
import numpy as np
from Datasets.GasDrift import dataset

# Parameters used by the baseline paper
# iter = 10  # Every selected solution is iterated by iter = 10
# Pmax = 20  # In each iteration Pmax = 20
# p = 0.5    # Probability of changing neurons at the particular layer
# K = 0.03   # The updation of neurons (K) is set to 3%
# maxim = 5  # Maximum amount of hidden layers
dataset_train, dataset_test = dataset(return_tensor=True)
inicio = time()
best_Gas, lista_melhores = TabuSearchForDFNN.tabu_search(128, 6, dataset_train, maxim=5, iter=10)
print("\nBest results for Gas-Drift:")
for i in best_Gas:
  print("Calculating the fitness of HL={} and HN={}".format(i.HL, i.N_List))
  print('Error Valid: {:.6f}'.format(i.error_valid))
  print('Standard deviation Valid: {:.6f}'.format(i.std_valid))
  print("Testing...")
  train_acc, test_acc = TabuSearchForDFNN.test_model(i, 128, 6, dataset_train, dataset_test)
  print('Average Acc Train:', np.mean(train_acc))
  print('Standard deviation Train:', np.std(train_acc))
  print('Average Acc Test:', np.mean(test_acc))
  print('Standard deviation Test:', np.std(test_acc))
print("\nTotal model time: {:.5f} minutes".format((time()-inicio)/60))

# Save the best information for plotting charts
List_HL = [i.HL for i in lista_melhores]
List_N_List = [i.N_List for i in lista_melhores]
List_e_v = [i.error_valid for i in lista_melhores]
List_d_v = [i.std_valid for i in lista_melhores]
print("\nList_HL =", List_HL, "\nList_N_List =", List_N_List, "\nList_e_v =", List_e_v, "\nList_d_v =", List_d_v)
print()