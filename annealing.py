from adjacency import AdjMatrix, adjmatrix_to_edges, pretty_vertices
from genetic import random_solution, genome_fitness, mutate
from path import PathMatrix
from results import Result, StepResult
import math
import random

def sa_loop(am: AdjMatrix, k_postmen: int, temperature: float, alpha: float, loops: int, start_idx: int, debug=True):
    """Main SA loop

    :param am: _description_
    :type am: AdjMatrix
    :param k_postmen: _description_
    :type k_postmen: int
    :param start_idx: _description_
    :type start_idx: int
    """

    result = Result(algorithm="SA", graph_name="")
    edges = adjmatrix_to_edges(am)
    pm = PathMatrix(am)
    cur_sol = random_solution(edges, k_postmen)
    cur_fit, cur_costs, cur_vertices = genome_fitness(cur_sol, pm, start_idx)

    temp = temperature
    for _ in range(loops):
        new_sol = mutate(cur_sol)
        # current value of fitness here is the cost, we want to minimize it
        new_fit, new_costs, new_vertices = genome_fitness(new_sol, pm, start_idx)
        if new_fit <= cur_fit:
            cur_sol = new_sol
            cur_fit = new_fit
            cur_costs = new_costs
            cur_vertices = new_vertices
            temp = temp * alpha
        else:
            p = math.exp((cur_fit-new_fit)/temp) # probability of accepting worse solution, note that the value in exp is negative
            if debug:
                print(p)
            r = random.random()
            if r <= p:
                cur_sol = new_sol
                cur_fit = new_fit
                cur_costs = new_costs
                cur_vertices = new_vertices
            temp = temp * alpha
        if debug:
            pretty_paths = [pretty_vertices(p) for p in cur_vertices]
            print(f'best solution costs {cur_costs} genome {cur_sol[0]} paths: {pretty_paths} ')
        sr = StepResult(best_solution=cur_sol, costs=cur_costs, path_vertices=cur_vertices, pop_result=None)
        result.add_iteration(sr)

    return result