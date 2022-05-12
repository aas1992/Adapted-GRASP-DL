import random
import copy
from Models import tabnet


# Class that creates a new individual
class Individual:
    def __init__(self, N_d, N_a, l_r, N_steps):
        self.N_d = N_d
        self.N_a = N_a
        self.l_r = l_r
        self.N_steps = N_steps
        self.error_valid = None
        self.std_valid = None

def repeat_remove(C):
    C_x = []
    N_list = []
    for i in C:
        if [i.N_d, i.N_a, i.l_r] not in N_list:
            C_x.append(i)
            N_list.append([i.N_d, i.N_a, i.l_r])
    return C_x

def grasp(Xtrain, ytrain, max=5, iter=10):
    Optimal = []  # Returns the best solution for each new hidden layer added
    Solucoes = []  # Returns all candidate solutions for the best solution

    for N_steps in range(1, max+1): # A "for" for maximum architectural steps: N_steps
        print("Initial Iteration")
        E = []
        parcial_solutions = {}  # Save the partial solutions from the increment of each element c(e))
        complete_solutions = {} # Save the complete solutions that were run with cross validation)

        s_linha, parcial_solutions, E = greedy_construction(E, N_steps, parcial_solutions, Xtrain, ytrain)
        s_best, complete_solutions = fitness_solution(s_linha, complete_solutions, Xtrain, ytrain)

        # s_linha is best from neighbor of s[x]
        s_l, complete_solutions = local_search(s_best, complete_solutions, Xtrain, ytrain) # calling ALgorithm 2

        # Evaluate the solutions
        if s_l.error_valid < s_best.error_valid:
            s_best = s_l

        for x in range(1, iter):
            print("\nIteration:", x)
            # Semi-greedy construction
            s_linha, parcial_solutions, E = greedy_construction(E, N_steps, parcial_solutions, Xtrain, ytrain)

            # Create the complete solution for the choice of greedy_construction
            s_a, complete_solutions = fitness_solution(s_linha, complete_solutions, Xtrain, ytrain)

            # s_linha is best from neighbor of s[x]
            s_b, complete_solutions = local_search(s_a, complete_solutions, Xtrain, ytrain)

            # Evaluate the solutions
            if s_b.error_valid < s_best.error_valid:
                s_best = s_b
        Solucoes.append(complete_solutions)      # complete_solutions Soluções contém todas as soluções encontradas para gerar gráficos de evolução da heurística
        Optimal.append(s_best)  # List contains best architecture with minimum testing error
    return Optimal, Solucoes  # or optimal[-1]

def greedy_construction(E, N_steps, parcial_solutions, Xtrain, ytrain, alfa=0.2):
    # Greedy_Randomized_Construction
    # Initialize the candidate set: C <- E
    for i in range(0, 8):
        # Create the initial solution
        a, b = 8, 65  # Minimum and maximum values for Widths
        N_d = random.randint(a, b)  # N_d receive the width of the decision prediction layer.
        N_a = random.randint(a, b)  # N_a receive the Width of the attention embedding for each mask
        l_r = random.choice([0.02, 0.03, 0.04])  # Learning rate
        # 1. Increase the structure in s_0
        E.append(Individual(N_d, N_a, l_r, N_steps))
    C = repeat_remove(E)
    for c in C:
        if (c.N_d, c.N_a, c.l_r) in parcial_solutions:
            c.error_valid = parcial_solutions[(c.N_d, c.N_a, c.l_r)]
        else:
            c.error_valid = tabnet.greedy_function(c, Xtrain, ytrain)
            parcial_solutions[(c.N_d, c.N_a, c.l_r)] = c.error_valid

    C.sort(key = lambda x :x.error_valid)
    c_min = C[0].error_valid
    c_max = C[1].error_valid
    # RCL
    RCL = [c for c in C if (c.error_valid <= c_min + alfa*(c_max-c_min))]
    # randomly select c* de RCL:
    c_best = random.choice(RCL)
    # Add c_best to partial solution
    _ = C.pop(C.index(c_best))
    E = copy.deepcopy(C)
    print("\nBest Construction Greedy")
    print("N_d={}, N_a={} e l_r={}:".format(S.N_d, S.N_a, S.l_r))
    print('Error valid:', c_best.error_valid)
    return c_best, parcial_solutions, E

def fitness_solution(solution, complete_solutions, Xtrain, ytrain):
    if (solution.N_d, solution.N_a, solution.l_r) in complete_solutions:
        solution.error_valid, solution.std_valid = complete_solutions[(solution.N_d, solution.N_a, solution.l_r)]
    else:
        solution.error_valid, solution.std_valid = tabnet.fitness_function(solution, Xtrain, ytrain)
        complete_solutions[(solution.N_d, solution.N_a, solution.l_r)] = (solution.error_valid, solution.std_valid)
    return solution, complete_solutions


def local_search(solution, complete_solutions, Xtrain, ytrain, Pmax=10, p=0.5):
    print("\nLocal Search:")
    candidate_List = [None for index in range(Pmax)]  # Candidate list is create with null values
    # For the first half of Pmax
    print("Busca local")
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
            s_neighbor.l_r += s_neighbor.l_r * 0.1
            s_neighbor.l_r = round(s_neighbor.l_r, 6)

        if (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r) in complete_solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = complete_solutions[
                (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = tabnet.fitness_function(s_neighbor, Xtrain,
                                                                                   ytrain)  # Using GDM with momentum 0.7
            complete_solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)] = (
            s_neighbor.error_valid, s_neighbor.std_valid)

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
            s_neighbor.l_r -= s_neighbor.l_r * 0.1
            s_neighbor.l_r = round(s_neighbor.l_r, 6)

        if (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r) in complete_solutions:
            s_neighbor.error_valid, s_neighbor.std_valid = complete_solutions[
                (s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)]
        else:
            s_neighbor.error_valid, s_neighbor.std_valid = tabnet.fitness_function(s_neighbor, Xtrain,
                                                                                   ytrain)  # Using GDM with momentum 0.7
            complete_solutions[(s_neighbor.N_d, s_neighbor.N_a, s_neighbor.l_r)] = (
            s_neighbor.error_valid, s_neighbor.std_valid)

        candidate_List[i] = s_neighbor  # Update Ft of candidate_List[i]

    # Update Ft of candidate_List[i+(Pmax/2)] # candidate_List[i] is collection of HL, N_List[], training error and testing error
    candidate_List.sort(key=lambda x: x.error_valid)  # Sort neighbors by fitness value
    return candidate_List[0], complete_solutions  # return best solution of candidate List
