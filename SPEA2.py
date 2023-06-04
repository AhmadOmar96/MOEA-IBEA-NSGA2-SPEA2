from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT6
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization.plotting import Plot
import time

problem = ZDT6()

max_evaluations = 2500

algorithm = SPEA2(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability=0.5, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

start_time = time.time()
algorithm.run()
solutions = algorithm.get_result()
front = get_non_dominated_solutions(solutions)
end_time = time.time() - start_time
print("--- %s seconds ---" % end_time)

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='SPEA2-ZDT6')