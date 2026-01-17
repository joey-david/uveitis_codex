from mvcavit.pso import PSO


def test_pso_optimizes_simple_function():
    pso = PSO(dim=2, particles=4, seed=1)

    def fitness(weights):
        return abs(weights[0] - 0.7) + abs(weights[1] - 0.3)

    weights = pso.optimize(fitness, iters=3)
    assert abs(weights[0] + weights[1] - 1.0) < 1e-6
