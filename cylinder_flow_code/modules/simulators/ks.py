import numpy as np

n_cinps = 3
n_states = 60
n_outputs = 3
dt = 0.025
nskip = 400
dx = 1
V = 0.2
R = 0.25
P = 0.05
Q = 0.0005

x = np.arange(n_states)

def gaussian_profile(center, sigma):
    return np.exp(-((x - center) / sigma) ** 2) / sigma

Bu = np.stack([
    gaussian_profile(9, 4),
    gaussian_profile(29, 4),
    gaussian_profile(49, 4)
])

def derivatives(cinps, u):
    
    u_nl = (u ** 2) / 2

    u_nl_m2 = np.roll(u_nl, 2)
    u_nl_m1 = np.roll(u_nl, 1)
    u_nl_p1 = np.roll(u_nl, -1)
    u_nl_p2 = np.roll(u_nl, -2)
    u1 = (u_nl_m2 - 8*u_nl_m1 + 8*u_nl_p1 - u_nl_p2) / (12 * dx)

    u_m2 = np.roll(u, 2)
    u_m1 = np.roll(u, 1)
    u_p1 = np.roll(u, -1)
    u_p2 = np.roll(u, -2)
    u2 = (-u_m2 + 16*u_m1 - 30*u + 16*u_p1 - u_p2) / (12 * dx**2)

    u_m3 = np.roll(u, 3)
    u_p3 = np.roll(u, -3)
    u4 = (-u_m3 + 12*u_m2 - 39*u_m1 + 56*u - 39*u_p1 + 12*u_p2 - u_p3) / (6 * dx**4)

    ut = -u1 - (P*u2 + u4) / R - Q * np.mean(u - V)
    return ut

def sim(cinps, states):
    cur_states = states.copy()

    total_profile = dt * np.tensordot(cinps, Bu, axes=(0, 0))

    for _ in range(nskip):
        f1 = dt * derivatives(cinps, cur_states)
        f2 = dt * derivatives(cinps, cur_states + f1 / 2)
        f3 = dt * derivatives(cinps, cur_states + f2 / 2)
        f4 = dt * derivatives(cinps, cur_states + f3)
        cur_states += (f1 + 2*f2 + 2*f3 + f4) / 6
        cur_states += total_profile

    return cur_states, get_sensor_data(cur_states)

def get_sensor_data(states):
    return states[::20]