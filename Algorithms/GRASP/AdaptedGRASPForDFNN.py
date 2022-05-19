import torch
import random
import copy
import numpy as np
import torch.utils.data as data_utils
from sklearn.model_selection import KFold, train_test_split
from Models import dfnn


def seed_everything(seed=1062):
  np.random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

# Class that creates a new individual
class Individual:
    def __init__(self, HL, N_List):
        self.HL = HL
        self.N_List = N_List
        self.error_valid = None
        self.std_valid = None

def repeat_remove(C):
    C_x = []
    N_List_list = []
    for i in C:
        if i.N_List not in N_List_list:
            C_x.append(i)
            N_List_list.append(i.N_List)
    return C_x

def test_model(solution, input, output, dataset, dataset_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(1006)

    train_loader = data_utils.DataLoader(dataset, batch_size=100, shuffle=True)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=100, shuffle=True)

    model = dfnn.Feedforward(input, solution.HL,
                             solution.N_List,
                             output)  # input_size, hidden_size, N_List, output_size .to(device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_acc, test_acc = [], []

    # Training
    for epoch in range(90):  # Epochs
        for x_train, y_train in train_loader:
            optimizer.zero_grad()  # Zero gradient buffers
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_train.to(device))  # Pass data though the net work
            loss = criterion(y_pred, y_train.to(device))  # Compute Loss
            loss.backward(retain_graph=True)  # Backpropagate
            optimizer.step()  # Update weights

    # Testing Train dataset
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc.append(100 * correct / total)

    # Testing Test dataset
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc.append(100 * correct / total)
    return train_acc, test_acc


def fitness_fuction(solution, input, output, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(1006)
    print("Calculating the fitness of HL={} and HN={}".format(solution.HL, solution.N_List))

    results_valid = []  # For fold results
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # random_state=42 Define the K-fold Cross Validator

    # K-fold Cross Validation model evaluation
    for train_ids, test_ids in kfold.split(dataset):  # Xtrain, ytrain
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                                  sampler=torch.utils.data.SubsetRandomSampler(train_ids))
        testloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                                 sampler=torch.utils.data.SubsetRandomSampler(test_ids))
        model = dfnn.Feedforward(input,
                                 solution.HL,
                                 solution.N_List,
                                 output)  # input_size, hidden_size, N_List, output_size .to(device)
        model.to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(90):  # Run the training loop for defined number of epochs
            for inputs, targets in trainloader:  # Iterate over the DataLoader for training data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward(retain_graph=True)
                optimizer.step()

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:  # Iterate over the test data and generate predictions
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()  # tráz de volta para a CPU
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            results_valid.append(100.0 * (correct / total))

    valid_average = np.mean(results_valid)
    valid_std = np.std(results_valid)
    print("Error Accuracy:", 100 - valid_average)
    print("Std. Dev. Accuracy:", valid_std)
    return 100 - valid_average, valid_std


def greedy_function(solution, input, output, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Calculating g(c) of HL={} and HN={}".format(solution.HL, solution.N_List))
    seed_everything(1006)
    results_valid = []

    X_train, X_valid, y_train, y_valid = train_test_split(dataset.tensors[0], dataset.tensors[1], test_size=.10,
                                                          random_state=1, shuffle=True)
    dataset = data_utils.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))  # Converting to Tensor
    dataset_valid = data_utils.TensorDataset(torch.FloatTensor(X_valid),
                                             torch.LongTensor(y_valid))  # Converting to Tensor
    trainloader = data_utils.DataLoader(dataset, batch_size=100, shuffle=True)
    testloader = data_utils.DataLoader(dataset_valid, batch_size=100, shuffle=True)

    model = dfnn.Feedforward(input,
                             solution.HL,
                             solution.N_List,
                             output)  # input_size, hidden_size, N_List, output_size .to(device)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(90):  # Run the training loop for defined number of epochs
        for inputs, targets in trainloader:  # Iterate over the DataLoader for training data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            results_valid.append(100.0 * (correct / total))

    valid_average = np.max(results_valid)
    print("Error accuracy:", 100 - valid_average)
    return 100 - valid_average


def grasp(input, output, dataset, maxim=2, iter=10):
    Optimal = []  # Returns the best solution for each new hidden layer added
    Solucoes = []  # Returns all candidate solutions for the best solution
    for HL in range(2, maxim + 1):
        print("\Starting with HL={}".format(HL))
        print("Iteration 01")
        E = []
        parcial_solutions = {}  # Save the partial solutions from the increment of each element c(e))
        complete_solutions = {}  # Save the complete solutions that were run with cross validation)

        s_linha, parcial_solutions, E = greedy_construction(E, HL, parcial_solutions, input, output, dataset)
        s_best, complete_solutions = fitness_solution(s_linha, complete_solutions, input, output, dataset)

        # s_l is best from neighbor of s[x]
        s_l, complete_solutions = local_search(s_best, complete_solutions, input, output, dataset)

        # Evaluate the solutions
        if s_l.error_valid < s_best.error_valid:
            s_best = s_l

        for x in range(0, iter - 1):
            print("\nIteration:", x + 2)
            # Semi-greedy construction
            s_linha, parcial_solutions, E = greedy_construction(E, HL, parcial_solutions, input, output, dataset)

            # Create the complete solution for the choice of greedy_construction
            s_a, complete_solutions = fitness_solution(s_linha, complete_solutions, input, output, dataset)

            # s_b is best from neighbor of s[x]
            s_b, complete_solutions = local_search(s_a, complete_solutions, input, output, dataset)
            # Evaluate the solutions
            if s_b.error_valid < s_best.error_valid:
                s_best = s_b
        Solucoes.append(complete_solutions)
        Optimal.append(s_best)
    return Optimal, Solucoes


def greedy_construction(E, HL, parcial_solutions, input, output, dataset, alfa=0.2):
    # Greedy_Randomized_Construction
    for i in range(0, 8):  # Coloquei pra iniciar com 8 candidatos
        # Inputs parameters
        I, O = input, output  # Input Neurons and Output Neurons
        N_List = []
        for HN in range(0, HL):  # Create the Initial Solution
            a, b = int((I + O) / 2), int(((I + O) * 2) / 3)
            N_List.append(random.randint(a, b))  # N_List receive the number of neurons in the hidden layer HL
            I = N_List[HN]
            E.append(Individual(HL, N_List))

    C = repeat_remove(E)
    # Evaluate the incremental cost c(e) for all e belongs C:
    for c in C:
        if (str(c.N_List)) in parcial_solutions:
            c.error_valid = parcial_solutions[str(c.N_List)]
        else:
            c.error_valid = greedy_function(c, input, output, dataset)  # Using GDM with momentum 0.7
            parcial_solutions[str(c.N_List)] = (c.error_valid)

    C.sort(key=lambda x: x.error_valid)
    c_min = C[0].error_valid
    c_max = C[-1].error_valid

    # Definir RCL
    RCL = [c for c in C if (c.error_valid <= c_min + alfa * (c_max - c_min))]

    c_best = random.choice(RCL)
    _ = C.pop(C.index(c_best))
    E = copy.deepcopy(C)
    return c_best, parcial_solutions, E


def fitness_solution(solution, complete_solutions, input, output, dataset):
    # Create a complete solution for the solution:
    if (str(solution.N_List)) in complete_solutions:
        solution.error_valid, solution.std_valid = complete_solutions[str(solution.N_List)]
    else:
        solution.error_valid, solution.std_valid = fitness_fuction(solution, input, output, dataset)
        complete_solutions[str(solution.N_List)] = (solution.error_valid, solution.std_valid)
    return solution, complete_solutions


def local_search(solution, complete_solutions, input, output, dataset, Pmax=10, p=0.5, K=0.03):
    """
    A local search of DFNN model hyperparameters
    """
    print("\nLocal Search:")
    # Initialize
    candidate_List = [None for _ in range(Pmax)]  # Candidate list is create with null values
    for i in range(0, int(Pmax / 2)):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors

        # Case 1
        for j in range(0, solution.HL):
            if random.random() >= p:  # p is probability
                s_neighbor.N_List[j] = round(s_neighbor.N_List[j] + s_neighbor.N_List[j] * K)

        if (str(s_neighbor.N_List)) in complete_solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = complete_solutions[str(s_neighbor.N_List)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = fitness_fuction(s_neighbor, input, output, dataset)
            complete_solutions[str(s_neighbor.N_List)] = (s_neighbor.error_valid, s_neighbor.std_valid)
        candidate_List[i] = s_neighbor  # Update Ft of candidate_List[i]

    for i in range(int(Pmax / 2), Pmax):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors
        # Case 2
        for j in range(0, solution.HL):
            if random.random() >= p:  # p is probability
                s_neighbor.N_List[j] = round(s_neighbor.N_List[j] - s_neighbor.N_List[j] * K)

        if (str(s_neighbor.N_List)) in complete_solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = complete_solutions[str(s_neighbor.N_List)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = fitness_fuction(s_neighbor, input, output, dataset)
            complete_solutions[str(s_neighbor.N_List)] = (s_neighbor.error_valid, s_neighbor.std_valid)
        candidate_List[i] = s_neighbor

    candidate_List.sort(key=lambda x: x.error_valid)  # Sort neighbors by fitness value
    return candidate_List[0], complete_solutions  # return best solution of candidate List
