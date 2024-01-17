import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
        k_cr_list.append(k)

    # Plot the hodograph for one specific T and k_cr
    chosen_T = T_list[0]
    chosen_k_cr = k_cr_list[0]

    B_chosen = [chosen_k_cr]
    A_chosen = [T1 * chosen_T, (T1 + chosen_T), 1, chosen_k_cr]
    sys_chosen = signal.TransferFunction(B_chosen, A_chosen)

    w, GM_chosen, phase_chosen = signal.bode(sys_chosen, np.logspace(-3, 1, 1000))
    U_chosen = GM_chosen * np.cos(np.radians(phase_chosen))
    V_chosen = GM_chosen * np.sin(np.radians(phase_chosen))

    plt.figure()
    plt.plot(U_chosen, V_chosen, label=f'Hodograph for T={chosen_T}, k_cr={chosen_k_cr}')
    plt.plot(0, 0, 'r+', label='Origin', markersize=10)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Hodograph for T and k_cr')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the chosen T and k_cr values
    return T_list, k_cr_list

def k_dependency(T_list, k_cr_list):
    # Plot the graph of k_cr(T)
    plt.figure(figsize=(8, 6))
    plt.plot(T_list, k_cr_list, 'r-', linewidth=2)
    plt.xlabel('T')
    plt.ylabel('$K_{cr}$')
    plt.title('$K_{cr}$ as a function of $T$')
    plt.grid(True)
    plt.show()

def main():
    T1 = 0.6

    # Call the first function to find k_cr and T and plot hodograph
    chosen_T, chosen_k_cr = hodograph(T1)

    print("T\t\tK_cr\n")
    for i in range(1, len(chosen_T), 10):
        print(f'{chosen_T[i]:.1f}\t|\t{chosen_k_cr[i]:.2f}')

    # Call the second function to plot k_cr(T)
    k_dependency(chosen_T, chosen_k_cr)

if __name__ == "__main__":
    main()
