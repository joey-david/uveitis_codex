import numpy as np


def _softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


class PSO:
    def __init__(self, dim=2, particles=8, inertia=0.6, c1=1.4, c2=1.4, seed=42):
        self.dim = dim
        self.particles = particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)
        self.positions = self.rng.normal(size=(particles, dim))
        self.velocities = self.rng.normal(scale=0.1, size=(particles, dim))
        self.pbest = self.positions.copy()
        self.pbest_score = np.full((particles,), np.inf)
        self.gbest = self.positions[0].copy()
        self.gbest_score = np.inf

    def optimize(self, fitness_fn, iters=5):
        for _ in range(iters):
            for i in range(self.particles):
                weights = _softmax(self.positions[i])
                score = fitness_fn(weights)
                if score < self.pbest_score[i]:
                    self.pbest_score[i] = score
                    self.pbest[i] = self.positions[i]
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.positions[i]
            r1 = self.rng.random(size=self.positions.shape)
            r2 = self.rng.random(size=self.positions.shape)
            cognitive = self.c1 * r1 * (self.pbest - self.positions)
            social = self.c2 * r2 * (self.gbest - self.positions)
            self.velocities = self.inertia * self.velocities + cognitive + social
            self.positions = self.positions + self.velocities
        return _softmax(self.gbest)
