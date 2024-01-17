import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random

def hodograph(T1):
    # Range of T values
    T_values = np.linspace(0.1, 5, 100)

    # Initialize lists to store T and k_cr values
    T_list = []
    k_cr_list = []

    for T in T_values:
        # Create the transfer function
        B = [1]  # The value of k doesn't matter here, as it will be adjusted later
        A = [T1 * T, (T1 + T), 1, 1]
        sys = signal.TransferFunction(B, A)

        # Frequency range for hodograph
        w, GM, _ = signal.bode(sys, np.logspace(-3, 10, 1000))

        # Find the index of the frequency where magnitude is closest to 1
        index_closest_to_1 = np.argmin(np.abs(GM - 1.0))
        frequency_closest_to_1 = w[index_closest_to_1]

        # Find the corresponding k value by adjusting the transfer function
        k = 1 / max(GM)

        # Store T and k_cr values
        T_list.append(T)
        k_cr_list.append(k+1)

    # Plot the hodograph for one specific T and k_cr
    chosen_T = T_list[49]
    chosen_k_cr = k_cr_list[49]

    B_chosen = [chosen_k_cr]
    A_chosen = [T1 * chosen_T, (T1 + chosen_T), 1, chosen_k_cr]
    sys_chosen = signal.TransferFunction(B_chosen, A_chosen)

    w, GM_chosen, phase_chosen = signal.bode(sys_chosen, np.logspace(-3, 1, 1000))
    U_chosen = GM_chosen * np.cos(np.radians(phase_chosen))
    V_chosen = GM_chosen * np.sin(np.radians(phase_chosen))

    print("T\t\tK_cr\n")
    for i in range(1, len(T_list), 10):
        print(f'{T_list[i]:.1f}\t|\t{k_cr_list[i]:.2f}')

    fig, graph = plt.subplots(1, 2, figsize=(12, 6))
    graph[0].plot(U_chosen, V_chosen, label=(f'Годограф для T={chosen_T:.1f}, '+'$K_{cr}$'+f' ={chosen_k_cr:.3f}'))
    graph[0].plot(0, 0, 'r+', label='Точка (0, 0)', markersize=10)
    graph[0].set_xlabel('Real')
    graph[0].set_ylabel('Imaginary')
    graph[0].set_title('Годограф для T и $K_{cr}$')
    graph[0].legend()
    graph[0].grid(True)

    graph[1].plot(T_list, k_cr_list, 'r-', linewidth=2)
    graph[1].set_xlabel('T')
    graph[1].set_ylabel('$K_{cr}$')
    graph[1].set_title('$K_{cr}$ как функция от аргумента $T$')
    graph[1].grid(True)
    plt.show()

    # Return the chosen T and k_cr values
    return T_list, k_cr_list

def graphs(W, A):
    # Step Response
    plt.subplot(3, 2, 1)
    t, y = signal.step(W)
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Step Response')
    plt.title('Step Response')
    plt.grid(True)

    # Impulse Response
    plt.subplot(3, 2, 2)
    t, y = signal.impulse(W)
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Impulse Response')
    plt.title('Impulse Response')
    plt.grid(True)

    # Bode Plot - Magnitude
    plt.subplot(3, 2, 3)
    w, mag, phase = signal.bode(W)
    plt.semilogx(w, mag)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.title('Bode Plot - Magnitude')
    plt.grid(True)

    # Bode Plot - Phase
    plt.subplot(3, 2, 4)
    plt.semilogx(w, phase)
    plt.xlabel('Frequency')
    plt.ylabel('Phase (degrees)')
    plt.title('Bode Plot - Phase')
    plt.grid(True)

    # Nyquist Plot
    plt.subplot(3, 2, 5)
    w, h = signal.freqresp(W)
    plt.plot(h.real, h.imag)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Nyquist Plot')
    plt.grid(True)

    # Pole-Zero Map
    plt.subplot(3, 2, 6)
    poles = np.roots(A)
    plt.plot(poles.real, poles.imag, 'x')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Pole-Zero Map')
    plt.grid(True)

    plt.tight_layout()  # Улучшает распределение графиков на изображении
    plt.show()

def main():
    T1 = 0.6
    delta_k = 0.1
    T_list, k_cr_list = hodograph(T1)

    T = T_list[99]
    k = k_cr_list[99]

    b_n_a = [([k - delta_k], [T1*T, T1 + T, 1, k - delta_k]),
            ([k + delta_k], [T1*T, T1 + T, 1, k + delta_k]),
            ([k], [T1*T, T1 + T, 1, k])]
    
    for i, (B_i, A_i) in enumerate(b_n_a):
        print(i+1)
        W_i = signal.TransferFunction(B_i, A_i)
        graphs(W_i, A_i)


if __name__ == "__main__":
    main()
