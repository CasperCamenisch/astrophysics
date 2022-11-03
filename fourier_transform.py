# Discrete Fourier transform

# Imports
import numpy as np
from matplotlib import pyplot as plt

# Fucnctions

def dft(array):
    """Computes the discrete fourier transform of a 1D array"""
    a_out = np.array([0+0j for x in array])
    N = len(array)
    for k, x in enumerate(array):
        x_out = 0
        for n, y in enumerate(array):
            x_out += np.exp(-1j * 2 * np.pi * n * k / N) * y
        a_out[k] = x_out

    return a_out



# Fourier transform tests

def fourier_transform_test():
    """X = [1, 2-1j, -1j, -1+2j]
    print(X)
    print(dft(X))"""


    func_1 = [np.exp(-np.pi * x**2) for x in np.linspace(0, 4*np.pi, num=100)]

    ffunc_1 = np.fft.fft(func_1)
    ffunc_2 = dft(func_1)

    plt.plot(func_1, label="original")
    plt.plot(ffunc_1, label="np fft")
    plt.plot(ffunc_2, label="manual dft")
    plt.legend()
    plt.show()

# Signal analysis

def signal_analysis():
    A = 1
    B = 1

    signal = [A * np.sin(2 * np.pi * 2 * x) + B * np.sin(2 * np.pi * 4 * x) for x in np.linspace(0, 10, num=1000)]

    fourier_transform = np.fft.fft(signal)

    freq = np.fft.fftfreq(len(fourier_transform))

    threshold = 0.5 * max(abs(fourier_transform))
    mask = abs(fourier_transform) > threshold
    peaks = freq[mask] * 10

    print(peaks)

    #inverse_fourier_transform = np.fft.ifft(fourier_transform)

    #TODO: how to get Fourier Transform Coefficients? library? np.fft.fftfreq()?

    #plt.plot(signal, label="signal")
    #plt.plot(fourier_transform, label="fft")
    plt.plot(freq, abs(fourier_transform), label="freq")
    #plt.legend()
    plt.show()

# Gibbs phenomenon

# Fourier transform and derivatives

# Fourier transform and convolution

# Poisson Equation: Density profile

# Poisson Equation: Fourier transform

# Poisson Equation: Comparison with the analytical solution


# main

def main():
    #fourier_transform_test()
    signal_analysis()

# Run

if __name__ == "__main__":
    main()