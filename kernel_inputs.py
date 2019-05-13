import numpy as np
import projections

class InputY:
    def __init__(self, A):
        pass

    def evaluate(self, y):
        return y

class InputX(InputY):
    def __init__(self, A):
        super().__init__(A)
        self.cp = projections.ConvexProjection(A)

    def evaluate(self, y):
        x=self.cp.evaluate(y)
        return x

class InputPsi(InputY):
    def __init__(self, A):
        super().__init__(A)
        self.wbp = projections.WarpingBackProjection(A)

    def evaluate(self, y):
        psi=self.wbp.evaluate(y)
        return psi
