import numpy as np
import matplotlib.pyplot as plt

class Kalman_filter():
    def __init__(self, A, B, C, Ex, Ez):
        self.A = A
        self.B = B
        self.C = C
        self.Sigma = Ex.copy()
        self.Ex = Ex
        self.Ez = Ez
        
    def __call__(self, x_est_previous, control, measurement):
        x_pred = self.A @ x_est_previous + self.B @ control
        Sigma_pred = self.A @ self.Sigma @ (self.A.T) + self.Ex
        
        K = Sigma_pred @ self.C.T @ np.linalg.inv(self.C @ Sigma_pred @ C.T + Ez)
        
        x_est = x_pred + K @ (measurement - self.C @ x_pred)
        self.Sigma = (np.eye(len(self.A)) - K @ self.C) @ Sigma_pred
        
        return x_est
    
# EXAMLE
    
# Generate noised data for material point under constant acceleration
    
if __name__ == '__main__':
    duration = 10
    dt = 0.1 # sample time
    u = 1 # input
    x = np.array([[0], [0]])


    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt**2], [dt]])
    C = np.array([[1, 0]])

    sigma_z = 50 # sensor variance
    Ez = sigma_z**2
    accel_noise_mag = 0.05 # acceleration disturbance
    Ex = accel_noise_mag * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])

    p = [] # position array
    v = [] # velocity array
    z = [] # measurement array
    t = []

    # generate data
    for i in np.arange(0, duration, dt):
        t.append(i)
        accel_noise = [[dt**2 / 2 * np.random.normal(0, accel_noise_mag)], [dt * np.random.normal(0, accel_noise_mag)]]
        x = np.matmul(A, x) + B*u + accel_noise
        meas_noise = np.random.normal(0, sigma_z)
        y = float(np.matmul(C, x)) + meas_noise
        
        p.append(x[0][0])
        v.append(x[1][0])
        z.append(y)
        
    # Kalman filter

    p_est = [] # position estimation array
    v_est = [] # velocity estimation array
    x = np.array([[0], [0]]) # initial state estimate

    kalman = Kalman_filter(A, B, C, Ex, Ez)

    for i in range(len(t)):
        x = kalman(x, np.array([[u]]), z[i])
        
        p_est.append(x[0][0])
        v_est.append(x[1][0])
        


    plt.plot(t, p, c='b', label='position')
    plt.plot(t, z, c='r', label='measurement')
    plt.plot(t, p_est, c='g', label='estimate')
    plt.legend()
    plt.xlabel('Time')
    plt.show()