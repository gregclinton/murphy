class Converge:
    def __init__(self, f, epsilon, maxsteps):
        self.k = 0
        self.f = f
        self.epsilon = epsilon
        self.mark = float('nan')
        self.maxsteps = maxsteps
                
    def done(self, theta):
        mark = self.f(theta)
        res = abs(mark - self.mark) < self.epsilon or self.k > self.maxsteps
        self.mark = mark
        self.k += 1
        return res