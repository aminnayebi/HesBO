import numpy as np

class SimpleEmbedding:
    def __init__(self, projection_matrix):
        self.A=projection_matrix

    def evaluate(self, low_dim_vector):
        # Multiplying A to y
        x = np.matmul(low_dim_vector, self.A)
        return x

class ConvexProjection(SimpleEmbedding):
    def evaluate(self, low_dim_vector):
        x=SimpleEmbedding.evaluate(self, low_dim_vector)
        # Projecting the values outside of X domain into the domain
        n = len(x)
        d = len(x[0])
        for i in range(n):
            for j in range(d):
                if x[i][j] > 1: x[i][j] = 1.0
                if x[i][j] < -1: x[i][j] = -1.0
        return x

class WarpingBackProjection(ConvexProjection):
    def __init__(self, projection_matrix):
        ConvexProjection.__init__(self,projection_matrix)
        org_bp_matrix = np.matmul(np.matmul(np.transpose(self.A), np.linalg.inv(np.matmul(self.A, np.transpose(self.A)))), self.A)
        self.bp_matrix=np.transpose(org_bp_matrix)

    def evaluate(self, low_dim_vector):
        x=ConvexProjection.evaluate(self, low_dim_vector)
        n = len(x)
        D = len(x[0])
        psi = np.empty((n, D))
        for i in range(n):
            suitable_for_bp = False
            for j in range(D):
                if x[i][j]==1.0 or x[i][j]==-1.0:
                    suitable_for_bp = True
            # Doing back projection for the points which are outside of the X domain
            if suitable_for_bp:
                z = np.matmul(x[i], self.bp_matrix)
                max_z_abs = max(np.absolute(z))
                z_prime = z / max_z_abs
                psi[i] = z_prime + np.linalg.norm(x[i] - z_prime, D) * (z_prime / np.linalg.norm(z_prime, D))
            else:
                psi[i] = np.copy(x[i])
        return psi
