import numpy as np

class SimpleGaussian:
    def __init__(self,effective_dim,main_dim):
        self.A = np.random.normal(0, 1, [effective_dim, main_dim])

    def evaluate(self):
        return self.A

class Normalized(SimpleGaussian):
    def evaluate(self):
        effective_dim=len(self.A)
        main_dim=len(self.A[0])
        new_matrix=np.zeros([effective_dim,main_dim])
        for i in range(effective_dim):
            norm=np.linalg.norm(self.A[i])
            new_matrix[i]=self.A[i]/norm
        return new_matrix

class Orthogonalized(SimpleGaussian):
    def evaluate(self):
        pass

