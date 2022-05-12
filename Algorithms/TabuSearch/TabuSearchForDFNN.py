import random
import copy
import torch # For Deep Learning
import torch.utils.data as data_utils
from sklearn.model_selection import KFold
from Models import dfnn
import numpy as np

def seed_everything(seed=1062):
  np.random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class Ssolution:
    def __init__(self, HL, N_List):
        self.HL = HL
        self.N_List = N_List
        self.error_valid = None
        self.std_valid = None
        self.tabu_tenure = 4

def test_model(solution, input, output, dataset, dataset_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(1006)

    train_loader = data_utils.DataLoader(dataset, batch_size=100, shuffle=True)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=100, shuffle=True)
    model = dfnn.Feedforward(input,
                             solution.HL,
                             solution.N_List,
                             output)  # input_size, hidden_size, N_List, output_size .to(device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_acc, test_acc = [], []

    # Training
    for epoch in range(90):  # Epochs
        # loss_train = list()
        for x_train, y_train in train_loader:
            optimizer.zero_grad()  # Zero gradient buffers

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_train.to(device))  # Pass data though the net work
            # Compute Loss
            loss = criterion(y_pred, y_train.to(device))  # y_pred.squeeze()  .to(device)
            loss.backward(retain_graph=True)  # Backpropagate
            optimizer.step()  # Update weights
            # loss_train.append(loss.item())

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


def calculate_fitness(solution, input, output, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(1006)
    print("\nCalculating the fitness of HL={} and HN={}".format(solution.HL, solution.N_List))
    acc_list = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # random_state=42 Define the K-fold Cross Validator
    # K-fold Cross Validation model evaluation
    for train_ids, test_ids in kfold.split(dataset):  # Xtrain, ytrain
        # Define data loaders for training and testing data in this fold
        trainloader = data_utils.DataLoader(dataset, batch_size=100,
                                            sampler=torch.utils.data.SubsetRandomSampler(train_ids))
        testloader = data_utils.DataLoader(dataset, batch_size=100,
                                           sampler=torch.utils.data.SubsetRandomSampler(test_ids))

        model = dfnn.Feedforward(input, solution.HL, solution.N_List,
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
                predicted = predicted.cpu()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            acc_list.append(100.0 * (correct / total))

    valid_average = np.mean(acc_list)
    valid_std = np.std(acc_list)
    print("Error Accuracy:", 100 - valid_average)
    print("Std. Dev. Accuracy:", valid_std)
    return 100 - valid_average, valid_std


def tabu_search(input, output, dataset, maxim=5, iter=10):
    optimal, Solucoes = [], []
    for HL in range(2, maxim + 1):
        print("\Starting with HL={}".format(HL))
        print("Iteration 01")
        # Input parameters
        I = input  # input neurons
        O = output  # output neurons
        N_List = []  #
        Tabu_List = []  # The Tabu list will add neurons considered tabu
        solutions = {}

        # Create the initial solution
        for HN in range(0, HL):
            a = int((I + O) / 2)
            b = int(((I + O) * 2) / 3)
            N_List.append(random.randint(a, b))  # N_List receive the number of neurons in the hidden layer HL
            I = N_List[HN]

        # 1. Incremento a estrtura em s0
        s_0 = Ssolution(HL, N_List)
        s_0.error_valid, s_0.std_valid = calculate_fitness(s_0, input, output, dataset)
        solutions[str(s_0.N_List)] = (s_0.error_valid, s_0.std_valid)
        S = []
        S.append(s_0)  # List to store the best solutions

        s_best = s_0  # s0 is the initial solution update with s_best
        Tabu_List.append(s_best)

        # 2. Generate neighbors and evaluate them
        for x in range(0, iter):
            print("\nIteration:", x + 2)
            s_linha, Solucoes = geneate_neighbor(S[x], solutions, input, output, dataset,
                                                 Solucoes)  # calling Algorithm 2
            # Decrease tabu_tenure of Tabu_List by one in last four solution
            if len(Tabu_List) >= 4:
                for i in range(-1, -5, -1):
                    Tabu_List[i].tabu_tenure -= 1
            else:
                for i in range(0, len(Tabu_List)):
                    Tabu_List[i].tabu_tenure -= 1

            if s_linha.error_valid < s_best.error_valid:
                Tabu_List.append(s_linha)
                s_best = s_linha
                S.append(s_linha)
            else:
                Tabu_List_NList = [s.N_List for s in Tabu_List]
                if (s_linha.N_List not in Tabu_List_NList) or (s_linha.N_List in Tabu_List_NList and Tabu_List[
                    Tabu_List_NList.index(s_linha.N_List)].tabu_tenure <= 0):
                    Tabu_List.append(s_linha)
                    S.append(s_linha)
                    Solucoes.append(s_linha)
                else:
                    S.append(s_linha)
                    Solucoes.append(s_linha)
        optimal.append(s_best)  # List contains best architecture with minimum testing error
        Solucoes.append(s_best)
    return optimal, Solucoes  # or optimal[-1]


def geneate_neighbor(solution, solutions, input, output, dataset, Solucoes, Pmax=20, p=0.5, K=0.03):
    # Initialize
    candidate_List = [None for _ in range(Pmax)]  # Candidate list is create with null values

    for i in range(0, int(Pmax / 2)):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors

        # Case 1
        for j in range(0, solution.HL):
            if random.random() >= p:  # p is probability
                s_neighbor.N_List[j] = round(s_neighbor.N_List[j] + s_neighbor.N_List[j] * K)
        if (str(s_neighbor.N_List)) in solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = solutions[str(s_neighbor.N_List)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = calculate_fitness(s_neighbor, input, output, dataset)
            solutions[str(s_neighbor.N_List)] = (s_neighbor.error_valid, s_neighbor.std_valid)
            Solucoes.append(s_neighbor)
        candidate_List[i] = s_neighbor  # Update Ft of candidate_List[i]

    for i in range(int(Pmax / 2), Pmax):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors

        # Case 2
        for j in range(0, solution.HL):
            if random.random() >= p:  # p is probability
                s_neighbor.N_List[j] = round(s_neighbor.N_List[j] - s_neighbor.N_List[j] * K)
        if (str(s_neighbor.N_List)) in solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = solutions[str(s_neighbor.N_List)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = calculate_fitness(s_neighbor, input, output, dataset)
            solutions[str(s_neighbor.N_List)] = (s_neighbor.error_valid, s_neighbor.std_valid)
            Solucoes.append(s_neighbor)
        candidate_List[i] = s_neighbor

        # Update Ft of candidate_List[i+(Pmax/2)] # candidate_List[i] is collection of HL, N_List[], training error and testing error
    candidate_List.sort(key=lambda x: x.error_valid)
    return candidate_List[0], Solucoes  # return best solution of candidate List and all solutions
