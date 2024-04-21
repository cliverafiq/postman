from tests import test_random_solution, test_genome_conversions
from path import test_dijkstra, test_pathmatrix
from genetic import ga_loop
from annealing import sa_loop
from onecpp import chinese_postman, adj_mat_to_netx_graph
from adjacency import AdjMatrix
import samples
from results import *
import networkx as nx
import matplotlib.pyplot as plt

DEBUG = False

def main():
    #test_random_solution()
    #test_genome_conversions()
    #test_dijkstra(samples.sample1)
    #test_pathmatrix(samples.sample1)
    am, idx = matrix_choice()
    task_choice(am, idx)
    return
    
    #sample = samples.sample1
    #sample = samples.gen_lattice((10,10), 10, 0).tolist()
    #sample = samples.gen_lattice((3,3), 10, 0).tolist()

    #run_1cpp_overhead(sample, 10)
    optimal_cost = chinese_postman(adj_mat_to_netx_graph(sample))
    print(f'1-CPP cost: {optimal_cost}')

    result = ga_loop(am=sample, k_postmen=5, pop_size=1000, fit_size=100, start_idx=0, loops=100, prob_mutation=0.05, debug=False)

    #result = sa_loop(am=sample, k_postmen=3, temperature=5000, alpha=0.999, loops=10000, start_idx=0, debug=True)
    #result = sa_loop(am=sample, k_postmen=3, temperature=5000, alpha=0.99, loops=1000, start_idx=0, debug=False)

    # plot_single_run_total_cost_vs_iter(result)
    plot_single_run_max_tour_and_total_cost_vs_iter(result)



def matrix_choice() -> Tuple[AdjMatrix, int]:
    method = int(input('1. Built-in sample\t 2. Lattice > '))
    start_idx = int(default_input("Start vertex index (0) > ", 0))
    if method == 1:
        return samples.sample1, start_idx
    elif method == 2:
        size = int(input('Enter number of vertices per side (e.g 5 for a 5x5 lattice) > '))
        return samples.gen_lattice((size, size), 10, 0), start_idx

def task_choice(am: AdjMatrix, start_idx: int):
    ch = [
        '1. 1-CPP optimal cost',
        '2. Genetic Algorithm custom run',
        '3. Simulated Annealing custom run',
        '4. 1-CPP: Optimal vs Genetic Algorithm',
        '5. 1-CPP: Optimal vs Simulated Annealing'  
          ]
    s = "\n".join(ch)
    s = s + "\n > "
    u_ch = int(input(s))
    if u_ch > len(ch) or u_ch < 1:
        print('invalid choice')
        return
    if u_ch == 1:
        optimal_cost = chinese_postman(adj_mat_to_netx_graph(am))
        print(f'1-CPP cost: {optimal_cost}')
    elif u_ch == 2:
        algo_fct = params_ga_choice(am, start_idx)
        result = algo_fct()
        plot_single_run_max_tour_and_total_cost_vs_iter(result)
    elif u_ch == 3:
        algo_fct = params_sa_choice(am, start_idx)
        result = algo_fct()
        plot_single_run_max_tour_and_total_cost_vs_iter(result)
    if u_ch == 4:
        algo_fct = lambda : ga_loop(am=am, k_postmen=1, pop_size=1000, fit_size=100, start_idx=start_idx, loops=100, prob_mutation=0.05, debug=DEBUG)
        run_1cpp_overhead(am, 10, algo_fct)
    elif u_ch == 5:
        algo_fct = lambda : sa_loop(am=am, k_postmen=1, temperature=5000, alpha=0.999, loops=10000, start_idx=start_idx, debug=DEBUG) 
        run_1cpp_overhead(am, 10, algo_fct)

def params_ga_choice(am: AdjMatrix, start_idx: int):
    k = int(default_input("number of postmen (3) > ", 3))
    pop_size = int(default_input("population size (1000) > ", 1000))
    fit_size = int(default_input("fit population size (100) > ", 100))
    loops = int(default_input("max loops (100) > ", 100))
    mutate = float(default_input("mutation probability (0.05) > ", 0.05))

    algo_fct = lambda : ga_loop(am=am, k_postmen=k, pop_size=pop_size, fit_size=fit_size, start_idx=start_idx, loops=loops, prob_mutation=mutate, debug=False) 
    return algo_fct

def params_sa_choice(am: AdjMatrix, start_idx: int):
    k = int(default_input("number of postmen (3) > ", 3))
    temp = int(default_input("temperature (5000) > ", 5000))
    alpha = float(default_input('alpha decay (0.99) > ', 0.99))
    loops = int(default_input("max loops (1000) > ", 1000))

    algo_fct = lambda : sa_loop(am=am, k_postmen=k, temperature=temp, alpha=alpha, loops=loops, start_idx=start_idx, debug=DEBUG)  
    return algo_fct

def default_input(prompt: str, default):
    in_str = input(prompt)
    return default if in_str == "" else in_str

def run_1cpp_overhead(adj_mat: AdjMatrix, loops: int, fct: Callable[[], Result]):
    optimal_cost = chinese_postman(adj_mat_to_netx_graph(adj_mat))
    print(f'1-CPP cost: {optimal_cost}')
    results = run_many(fct, loops)
    best_result, _, _ = find_best_step_in_results(results)
    plot_1cpp_cost_vs_algo(best_result, optimal_cost)


if __name__ == '__main__':
    main()
