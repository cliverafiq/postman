from tests import test_random_solution, test_genome_conversions
from path import test_dijkstra, test_pathmatrix
from genetic import ga_loop
from simulated import sa_loop
import samples

def genetic_algorithm():
    #test_random_solution()
    #test_genome_conversions()
    #test_dijkstra(samples.sample1)
    # test_pathmatrix(samples.sample1)

    ga_loop(am=samples.sample1, k_postmen=5, pop_size=5000, fit_size=100, start_idx=0, loops=100, prob_mutation=0.05)

def simmulated_annealing():

    sa_loop(am=samples.sample1, k_postmen=5, temperature=5000, alpha=0.99, start_idx=0, same_solution_max=150, same_cost_diff_max=15000)

user_choice = int(input("Do you want to do Genetic Algorithm (1) or Simulated Annealing (2) "))

if user_choice == 1:
    genetic_algorithm()

elif user_choice == 2:
    simmulated_annealing()

else:
    print("Incorrect input")
