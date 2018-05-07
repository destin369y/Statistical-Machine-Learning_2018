import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass

# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        # pass
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        u = np.random.uniform()
        v = np.random.uniform()
        z = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
        y = self.mu + z * self.sigma
        return y


# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        # pass
        self.Mu = Mu
        self.Sigma = Sigma
        self.A = np.linalg.cholesky(Sigma)

    def sample(self):
        u = np.random.uniform(size = self.Mu.shape)
        v = np.random.uniform(size = self.Mu.shape)
        z = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
        y = np.matmul(self.A, z) + self.Mu
        return y
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        #pass
        self.ap = ap

    def sample(self):
        cumsum = np.cumsum(self.ap)
        u = np.random.uniform()
        y = np.where(cumsum > u)[0][0]
        return y


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        pass


def main():
    un = UnivariateNormal(10, 1)
    y = [un.sample() for i in range(10000)]
    plt.hist(y, np.arange(5, 15, 0.5))
    plt.title('UnivariateNormal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    Mu = np.array([1, 1])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    mn = MultiVariateNormal(Mu, Sigma)
    z = [mn.sample() for i in range(10000)]
    x, y = zip(*z)
    plt.scatter(x, y)
    plt.title('MultiVariateNormal')
    plt.show()

    ap = [0.2, 0.4, 0.3, 0.1]
    ca = Categorical(ap)
    y = [ca.sample() for i in range(10000)]
    plt.hist(y)
    plt.title('Categorical')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
