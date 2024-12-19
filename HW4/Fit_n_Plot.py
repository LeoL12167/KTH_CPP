import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, lam):
    return np.exp(-lam * x)

def fit_and_plot():
    # Load Data
    path = r"C:\Users\leolu\OneDrive\KTH\SF2565 Program Construction in C++ for Scientific Computing\HW4_CPP_KTH\HW4\out\build\x64-debug\HW4\probabilities.csv"
    data = np.genfromtxt(path, delimiter=',')

    # Fit Data with initial guess for lam
    initial_guess = [1.0]
    popt, pcov = curve_fit(func, data[:,0], data[:,1], p0=initial_guess)
    # Plot Data
    plt.plot(data[:,0], data[:,1], 'b-', label='data')
    plt.plot(data[:,0], func(data[:,0], *popt), 'r-', label='fit: $\lambda$=%5.3f' % tuple(popt))


    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()
    print(popt)

fit_and_plot()