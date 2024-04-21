from dataclasses import dataclass
from typing import List, Any, Optional, Callable, Tuple
import matplotlib.pyplot as plt

Genome = List[Any]

@dataclass
class PopulationResult:
    best_fit: float
    worst_fit: float
    avg_fit: float
    best_fit_total_cost: float

@dataclass
class StepResult:
    best_solution: Genome
    costs: List[float]
    path_vertices: List[int]
    pop_result: Optional[PopulationResult]

class Result:
    def __init__(self, algorithm: str, graph_name: str) -> None:
        self.algo = algorithm
        self.graph_name = graph_name
        self.step_results: List[StepResult] = []

    def add_iteration(self, result: StepResult):
        self.step_results.append(result)

    def num_steps(self):
        return len(self.step_results)

    def total_costs(self):
        m = [sum(result.costs) for result in self.step_results]
        return m
    
    def max_tour_costs(self):
        m = [max(result.costs) for result in self.step_results]
        return m

    def best_step(self) -> Tuple[StepResult, int]:
        cur_best_fit = float('inf')
        cur_total = float('inf')
        cur_idx = 0
        cur_step = None
        for idx, step in enumerate(self.step_results):
            m = max(step.costs)
            if m < cur_best_fit:
                t = sum(step.costs)
                if t < cur_total:
                    cur_idx = idx
                    cur_step = step
                    cur_best_fit = m
                    cur_total = t
        return cur_step, cur_idx


def run_many(fct: Callable[[], Result], num: int) -> List[Result]:
    results = []
    for idx in range(num):
        print(f'running iteration {idx+1} out of {num}')
        results.append(fct())
    return results

def find_best_step_in_results(results: List[Result]) -> Tuple[Result, StepResult, int]:
    idx = 0
    cur_best_fit = float('inf')
    cur_total = float('inf')
    cur_step = None
    cur_iters = 0
    best_result = None
    for result in results:
        best_step, iters = result.best_step()
        m = max(best_step.costs)
        if m < cur_best_fit:
            t = sum(best_step.costs)
            if t < cur_total:
                cur_iters = iters
                cur_step = best_step
                cur_best_fit = m
                cur_total = t
                best_result = result
    return best_result, cur_step, cur_iters

def plot_1cpp_cost_vs_algo(result: Result, lower_bound: float):
    x = range(0, result.num_steps())
    costs = result.max_tour_costs()
    lows = [lower_bound] * result.num_steps()
    plt.plot(x, lows, label=f"Optimal 1-CPP route, cost: {lower_bound}")
    plt.plot(x, costs, label=f"Best {result.algo} route, cost: {min(costs)}")
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()


def plot_single_run_total_cost_vs_iter(result: Result):
    total_costs = result.total_costs()
    plt.plot(total_costs)
    plt.ylabel(f'Single {result.algo} run for {result.graph_name}')
    plt.show()

def plot_single_run_max_tour_and_total_cost_vs_iter(result: Result):
    x = range(0, result.num_steps())
    total_costs = result.total_costs()
    max_tour_costs = result.max_tour_costs()
    plt.plot(x, max_tour_costs, label="Max postman tour cost")
    plt.plot(x, total_costs, label="Total costs for all postmen")
    plt.ylabel(f'Single {result.algo} run for {result.graph_name}')
    plt.legend()
    plt.show()
