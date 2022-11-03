# Solving the Kepler equation numerically (in 2D)
# d^2r/dt^2 = -GM/|r^3| r       where r = (x, y)
# gives us the equation system
# dr/dt = v
# dv/dt = -GM*r / abs(r)^3

# Imports
import time

import numpy as np
from matplotlib import pyplot as plt


# Utility functions

def abs(p):
    return np.sqrt(p[0]**2 + p[1]**2)


# Step methods

def euler_forward(pn, h):
    pn1 = pn + h * kepler_eq(pn)
    return pn1


def euler_semiimplicit(pn, h):
    derivatives = kepler_eq(pn)
    # Define position and velocity separatly
    p0, v0 = pn[:2], pn[2:]
    dp, dv = derivatives[:2], derivatives[2:]
    # Simi-implicit method
    v1 = v0 + dv * h
    derivatives = kepler_eq(np.array([p0[0], p0[1], v1[0], v1[1]]))
    dp, dv = derivatives[:2], derivatives[2:]
    p1 = p0 + dp * h
    return np.array([p1[0], p1[1], v1[0], v1[1]])


def rk_mid(pn, h):
    pn1 = pn + h * kepler_eq(pn + (h/2) * kepler_eq(pn))
    return pn1


def rk_4(pn, h):
    k1 = h * kepler_eq(pn)
    k2 = h * kepler_eq(pn + (k1/2))
    k3 = h * kepler_eq(pn + (k2/2))
    k4 = h * kepler_eq(pn + k3)
    pn1 = pn + (k1/6) + (k2/3) + (k3/3) + (k4/6)
    return pn1


def leapfrog(pn, h):
    derivatives = kepler_eq(pn)
    # Define position and velocity separatly
    p0, v0 = pn[:2], pn[2:]
    dp, dv = derivatives[:2], derivatives[2:]

    # Leapfrog method
    v12 = v0 + dv * h / 2   # first half drift
    p1 = p0 + v12 * h       # kick

    # Recompute derivatives
    derivatives = kepler_eq(np.array([p1[0], p1[1], v12[0], v12[1]]))
    dp, dv = derivatives[:2], derivatives[2:]

    v1 = v12 + dv * h / 2  # second half drift

    return np.array([p1[0], p1[1], v1[0], v1[1]])


# Kepler equation, returning derivatives in on array

def kepler_eq(pn):
    # Parameters
    GM = 1
    # Derivatives
    dxdt = pn[2]
    dydt = pn[3]
    dvxdt = -GM * pn[0] / (abs(pn) ** 3)
    dvydt = -GM * pn[1] / (abs(pn) ** 3)

    return np.array([dxdt, dydt, dvxdt, dvydt])


# ODE solver

def solver(eccentricity, n_steps, step_method):
    # Initiate position and velocity
    pn = np.array([1, 0, 0, np.sqrt(1+eccentricity)])
    x_list = [pn[0]]
    y_list = [pn[1]]
    E_list = []
    L_list = []
    # Define step size
    h = 0.01

    # Main loop over each step
    for n in range(n_steps):
        # Perform update
        pn1 = step_method(pn, h)
        # Calculate orbital energy
        v = np.sqrt(pn1[2]**2 + pn1[3]**2)
        r = abs(pn1)
        E = v**2 / 2 + 1 / r
        # Calculate angular momentum
        Lz = pn1[0]*pn1[3] - pn1[1]*pn1[2]
        # Save positions and orbital energy/angular momentum
        x_list.append(pn1[0])
        y_list.append(pn1[1])
        E_list.append(E)
        L_list.append(Lz)
        # Update variable
        pn = pn1

    return x_list, y_list, E_list, L_list


def plot_methods():
    methods = [euler_forward, euler_semiimplicit, rk_mid, rk_4, leapfrog]

    fig, ax = plt.subplots(nrows=len(methods), ncols=3, figsize=(5 * len(methods), 15))
    for i, method in enumerate(methods):
        # Solve kepler equation
        x, y, E, L = solver(0.5, 50_000, method)
        # Plot
        ax[i, 0].set_title(str(method))
        ax[i, 0].plot(x, y)
        ax[i, 0].scatter(0, 0, color="red")
        ax[i, 0].axis("equal")
        ax[i, 1].plot(E)
        ax[i, 2].plot(L)

    plt.savefig("ODE_solvers_e05.png")


def plot_energy():
    methods = [euler_forward, euler_semiimplicit, rk_mid, rk_4, leapfrog]

    for i, method in enumerate(methods):
        # Solve kepler equation
        x, y, E, L = solver(0, 2_000, method)
        # Plot
        plt.plot(E, label=method)
        plt.legend()

    plt.savefig("orbital_energies_all.png")

def time_test():
    methods = [euler_forward, euler_semiimplicit, rk_mid, rk_4, leapfrog]

    for i, method in enumerate(methods):
        t0 = time.time()
        solver(0, 50_000, method)
        t1 = time.time()
        print(f"{method}\n{t1-t0} seconds for 50_000 steps")

# main function

def main():
    #plot_methods()
    #plot_energy()
    time_test()


if __name__ == "__main__":
    main()
