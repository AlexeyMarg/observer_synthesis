import numpy as np
import matplotlib.pyplot as plt


class particles_filter():
    def __init__(self, N_particles, state, update_state, sigma_meas, sigma_pos):
        self.N_particles = N_particles
        self.state = state
        self.sigma_meas = sigma_meas
        self.sigma_pos = sigma_pos
        self.update_state = update_state
        
        self.particles = np.zeros((N_particles, len(state)))
        for i in range(N_particles):
            for j in range(len(state)):
                self.particles[i][j] = state[j][0] + np.random.normal(self.sigma_pos[j][0])
                
        self.weights = np.ones(self.N_particles) / self.N_particles
        
    def prob(self, dist, meas_dist):
        return 1/np.sqrt(2 * np.pi * self.sigma_meas) * np.exp( -(dist - meas_dist)**2 / self.sigma_meas)
    
    def resample(self, particles, weights):
        newParticles = []
        newWeights = []
        N = len(particles)
        index = np.random.randint(0, N)
        betta = 0
        for i in range(N):
            betta = betta + np.random.uniform(0, 2*max(weights))
            while betta > weights[index]:
                betta = betta - weights[index]
                index = (index + 1)%N # индекс изменяется в цикле от 0 до N
            newParticles.append(particles[index])
            newWeights.append(weights[index])
        newWeights = newWeights / np.sum(newWeights)
        
        return np.array(newParticles), np.array(newWeights)
    
    def estimation(self, particles, weights):
        estimateX = np.zeros(len(self.state)) 
        for i in range(len(particles)):
            estimateX = estimateX + particles[i] * weights[i]
        return estimateX
        
    def __call__(self, control, measurement):
        new_particles = []
        for particle in self.particles:
            new_particles.append(self.update_state(particle, control))
        self.particles = new_particles
        
        for i in range(self.N_particles):
            self.weights[i] = self.prob(self.particles[i], measurement)
        self.weights = self.weights / np.sum(self.weights)
        
        self.particles, self.weights = self.resample(self.particles, self.weights)
        
        estX = self.estimation(self.particles, self.weights)
        
        return estX
    
    
# Example
if __name__ =='__main__':    
    
    sigma_pos = 1
    sigma_meas = 3

    robotX = 20 + np.random.normal(0, sigma_pos)

    def update_state(x, u):
        return x + u

    def movement(x, u):
        return x + u + np.random.normal(0, sigma_pos)

    def measurement(x):
        return x + np.random.normal(0, sigma_meas)
    
    filter = particles_filter(3000, np.array([[20]]), update_state,
                          np.array([[sigma_meas]]), np.array([[sigma_pos]]))
    
    u = [5, 10, 3, -7, 6, -10, 5, -6, 8, 4, -5]
    
    X = [robotX]
    Y = [measurement(robotX)]
    X_est = [measurement(robotX)]

    for step in range(len(u)):
        robotX = movement(robotX, u[step])
        robotMeas = measurement(robotX)
        estX = filter(u[step], robotMeas)
        X_est.append(float(estX))
        X.append(robotX)
        Y.append(robotMeas)
        
    plt.plot(range(len(X)), X)
    plt.plot(range(len(X_est)), X_est)
    plt.plot(range(len(Y)), Y)
    plt.show()