import random
import copy
from Models import tabnet


class Ssolution:
    def __init__(self, N_d, N_a, l_r, N_steps):
        self.N_d = N_d
        self.N_a = N_a
        self.l_r = l_r
        self.N_steps = N_steps
        self.error_valid = None
        self.std_valid = None
        self.tabu_tenure = 4

def TabuSearch(Xtrain, ytrain, max=5, iter=10):
    optimal, Solucoes = [], []  # The list of the best solutions
    for N_steps in range(1, max+1):  # A "for" for maximum architectural steps: N_steps
        print("\Starting with N_steps={}".format(N_steps))
        print("Iteration 01")
        Tabu_List = []  # The Tabu list will add neurons considered tabu
        solutions = {}
        # Create the initial solution
        a, b = 5, 65  # Minimum and maximum values for Widths
        N_d = random.randint(a, b)  # N_d receive the width of the decision prediction layer.
        N_a = random.randint(a, b)  # N_a receive the Width of the attention embedding for each mask
        l_r = 0.02  # Learning rate
        print('Starting with n_d={}, n_a={}, l_r={} and N_steps={}'.format(N_d, N_a, l_r, N_steps))

        # 1. Increase the structure in s0
        s_0 = Ssolution(N_d, N_a, l_r, N_steps)
        s_0.error_valid, s_0.std_valid = tabnet.fitness_function(s_0, Xtrain, ytrain)
        solutions[(s_0.N_d, s_0.N_a, s_0.l_r)] = (s_0.error_valid, s_0.std_valid)
        # s0, s_best, s_linha is structure of N_d, N_a, N_steps, training accuracy, validation error and testing error
        S = []
        S.append(s_0)  # List to store the best solutions

        s_best = s_0  # s0 is the initial solution update with s_best
        Tabu_List.append(s_best)

        # 2. Generate neighbors and evaluate them
        for x in range(0, iter):
            print("\nIteration:", x + 2)
            s_linha, Solucoes = geneate_neighbor(S[x], solutions, Xtrain, ytrain, Solucoes)  # calling Algorithm 2

            # Decrease tabu_tenure of Tabu_List by one in last four solution
            if len(Tabu_List) >= 4:
                for i in range(-1, -5, -1):
                    Tabu_List[i].tabu_tenure -= 1
            else:
                for i in range(0, len(Tabu_List)):
                    Tabu_List[i].tabu_tenure -= 1

            if s_linha.error_valid < s_best.error_valid:
                print("\nBest solution update:")
                print("N_d: {}, N_a: {}, L_r: {}, N_steps: {}, error_valid: {:.6f}".format(s_linha.N_d, s_linha.N_a,
                                                                                           s_linha.l_r, s_linha.N_steps,
                                                                                           s_linha.error_valid))
                Tabu_List.append(s_linha)
                s_best = s_linha
                S.append(s_linha)
            else:
                Tabu_List_N = [(s.N_d, s.N_a, s.l_r) for s in Tabu_List]
                if ((s_linha.N_d, s_linha.N_a, s_linha.l_r) not in Tabu_List_N) or (
                        (s_linha.N_d, s_linha.N_a, s_linha.l_r) in Tabu_List_N and Tabu_List[
                    Tabu_List_N.index((s_linha.N_d, s_linha.N_a, s_linha.l_r))].tabu_tenure <= 0):
                    Tabu_List.append(s_linha)
                    S.append(s_linha)
                    Solucoes.append(s_linha)
                else:
                    S.append(s_linha)
                    Solucoes.append(s_linha)
        optimal.append(s_best)  # List contains best architecture with minimum testing error
        Solucoes.append(s_best)
    return optimal, Solucoes  # or optimal[-1]


def geneate_neighbor(solution, solutions, Xtrain, ytrain, Solucoes, Pmax=10, p=0.5):
    # Initialize
    candidate_List = [None for index in range(Pmax)]  # Candidate list is create with null values

    # For the first half of Pmax
    for i in range(0, int(Pmax / 2)):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors

        # Case 1 (+)
        if random.random() >= p:  # p is probability
            s_neighbor.N_d += random.randint(1, 3)
            if random.random() >= p:  # p is probability
                s_neighbor.N_a += random.randint(1, 3)
        else:
            s_neighbor.N_a += random.randint(1, 3)
            if random.random() >= p:  # p is probability
                s_neighbor.N_d += random.randint(1, 3)
        if random.random() > p:
            s_neighbor.l_r += s_neighbor.l_r * 0.05
            s_neighbor.l_r = round(s_neighbor.l_r, 6)

        if (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r) in solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = tabnet.fitness_function(s_neighbor, Xtrain, ytrain)
            solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)] = (s_neighbor.error_valid, s_neighbor.std_valid)
            Solucoes.append(s_neighbor)
        candidate_List[i] = s_neighbor  # Update Ft of candidate_List[i]

    # For the second half of Pmax
    for i in range(int(Pmax / 2), Pmax):
        s_neighbor = copy.deepcopy(solution)  # Create a copy of solution to search the neighbors
        # Case 2 (-)
        if random.random() >= p:  # p is probability
            s_neighbor.N_d -= random.randint(1, 3)
            if random.random() >= p:  # p is probability
                s_neighbor.N_a -= random.randint(1, 3)
        else:
            s_neighbor.N_a -= random.randint(1, 3)
            if random.random() >= p:  # p is probability
                s_neighbor.N_d -= random.randint(1, 3)

        if random.random() > p:
            s_neighbor.l_r -= s_neighbor.l_r * 0.05
            s_neighbor.l_r = round(s_neighbor.l_r, 6)

        if (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r) in solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = tabnet.fitness_function(s_neighbor, Xtrain, ytrain)
            solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)] = (s_neighbor.error_valid, s_neighbor.std_valid)
            Solucoes.append(s_neighbor)
        candidate_List[i] = s_neighbor  # Update Ft of candidate_List[i]

    candidate_List.sort(key=lambda x: x.error_valid)
    return candidate_List[0], Solucoes  # return best solution of candidate List

