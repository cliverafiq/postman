from tests import test_random_solution, test_genome_conversions
from path import test_dijkstra, test_pathmatrix
from genetic import ga_loop
import samples

def main():
    #test_random_solution()
    #test_genome_conversions()
    #test_dijkstra(samples.sample1)
    # test_pathmatrix(samples.sample1)

    ga_loop(am=samples.sample1, k_postmen=3, pop_size=1000, fit_size=100, start_idx=0, loops=100, prob_mutation=0.05)

if __name__ == '__main__':
    main()
