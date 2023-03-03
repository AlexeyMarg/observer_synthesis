import numpy as np
import matplotlib.pyplot as plt

class extended_Kalman_filter():
    def __init__(self, g, h, dg_dx, dh_dx, Ex, Ez):
        self.g = g
        self.h = h
        self.dg_dx = dg_dx
        self.dh_dx = dh_dx
        self.Sigma = Ex.copy()
        self.Ex = Ex
        self.Ez = Ez
        
    def __call__(self, x_est_previous, control, measurement):
        Gt = self.dg_dx(x_est_previous, control)
        Ht = self.dh_dx(x_est_previous)
                
        x_pred = self.g(x_est_previous, control)
        Sigma_pred = Gt @ self.Sigma @ (Gt.T) + self.Ex
        
        K = Sigma_pred @ Ht.T @ np.linalg.inv(Ht @ Sigma_pred @ Ht.T + Ez)
        
        x_est = x_pred + K @ (measurement - self.h(x_pred))
        self.Sigma = (np.eye(len(x_est_previous))- K @ Ht) @ Sigma_pred
        
        return x_est
    
# EXAMPLE

# Plant model
# x_t = g(x_t-1, u_t) = Ax + Bu
# z_t = h(x_t) = Cx

if __name__ == '__main__':
    duration = 10
    dt = 0.1 # sample time
    u = 1 # input
    x = np.array([[0], [0]])


    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt**2], [dt]])
    C = np.array([[1, 0]])

    #g = Ax+Bu
    def g(x, u):
        return A @ x + B @ u

    def dg_dx(x, u):
        return A

    # h = Cx
    def h(x):
        return C

    def dh_dx(x):
        return C
    
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
        
    # Extended Kalman filter
    
    p_est = []
    v_est = []
    x = np.array([[0], [0]])

    ext_kalman = extended_Kalman_filter(g, h, dg_dx, dh_dx, Ex, Ez)

    for i in range(len(t)):
        x = ext_kalman(x, np.array([[u]]), z[i])
    
        p_est.append(x[0][0])
        v_est.append(x[1][0])
    
    plt.plot(t, p, c='b', label='position')
    plt.plot(t, z, c='r', label='measurement')
    plt.plot(t, p_est, c='g', label='estimate')
    plt.legend()
    plt.xlabel('Time')
    plt.show()