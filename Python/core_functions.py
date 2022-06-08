import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy

from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit import quantum_info as qi
from scipy.optimize import curve_fit
from scipy import sparse


def reconstruct_channel(cliffords, decay_rates01, decay_rates10, decay_rates11):
    """Reconstructs a two-qubit channel from local Clifford UIRS decay rates
    """

    reconstruction = np.zeros((16,16))

    for cliff in range(len(cliffords)):
        reconstruction += ((2**4 - 1) * (3**2 * decay_rates11[cliff] + 
                                                3 *  decay_rates01[cliff] + 
                                                3 *  decay_rates10[cliff] +
                                                1) - (2**4 - 2)) * np.real(qi.PTM(qi.Operator(np.conjugate(np.transpose(cliffords[cliff])))).data) / len(cliffords)

    return reconstruction


def sequence_correlation_function(ptms, A, number_qubits, number_samples, lengths, protocol_type, projection_matrix, initial_state=None, measurements=None):
    """
    ptms: list of lists with the same structure as sequences and circuits, but each gate transformed to a PTM
    A: probe superoperator (as PTM)
    number_qubits: number of qubits in the circuit (currently restricted to 2)
    number_samples: number of sampled circuits for each length
    lengths: list of circuit lengths
    protocol_type: can be "multi", "local01", "local10", or "local11" (depending on multi-qubit vs. local Clifford protocol and projection operators P_\omega)
    """
    
    # if initial_state or measurements is None: use all-zero initial state and computational basis measurements
    if initial_state is None:
        initial_state = get_initial_state_vec()
        
    if measurements is None:
        measurements = get_measurement_vecs()
    
    # normalization factors depending on protocol type:
    if protocol_type == 'multi':
        alpha = 2**number_qubits + 1 
    if protocol_type == 'local00':
        alpha = 2**number_qubits
    if protocol_type == 'local01' or protocol_type == 'local10':
        alpha = 2**number_qubits * 3
    if protocol_type == 'local11':
        alpha = 2**number_qubits * 3**2
    
    # get the PTM for B_x (initial state and measurement)
    B_list = []
    for povm in range(len(measurements)):
        B_matrix = projection_matrix @ initial_state @ measurements[povm]
        B_list.append(B_matrix)
    
    correlation_functions = np.empty((len(lengths), number_samples, 2**number_qubits))
    
    for i in range(len(lengths)): # loop over circuit lengths m
        for j in range(number_samples): # loop over number of samples N*K
            
            # first step: intersperse A into the sequences
            post_process_circ = intersperse(ptms[i][j], A)
            
            # get the matrix product of the gate PTMs with interspersed A:
            gates_matrix = np.identity(np.shape(A)[0])
            for gate in range(len(post_process_circ)): gates_matrix = projection_matrix @ post_process_circ[gate] @ projection_matrix @ gates_matrix
            
            for k in range(2**number_qubits): # loop over all possible outcomes
                # calculate the sequence correlation function (Eq. 12)
                correlation_functions[i][j][k] = alpha * np.trace(B_list[k] @ gates_matrix)
    
    return correlation_functions


def noisy_simulation(circuits, number_qubits, number_shots, number_samples, lengths, noise_model, gateset=None):
    """Use Qiskit to simulate the input circuits with noise, defined by the input noise_model
    """

    if gateset is None:
        gateset = ['rx', 'ry', 'rz', 'cx']
    
    noisy_sim = AerSimulator(shots = number_shots, noise_model=noise_model, basis_gates=gateset)

    # run the simulation:
    results = []
    
    for length in range(len(circuits)):
        
        result_list = []
        for rep in range(len(circuits[length])):

            transpiled_circ = transpile(circuits[length][rep], noisy_sim)
            result = noisy_sim.run(transpiled_circ).result()
            result_list.append(result)
            
        results.append(result_list)

    
    # extract the counts and calculate the corresponding probabilities:
    counts = np.empty((len(lengths), number_samples, 2**number_qubits))

    for i in range(len(lengths)): # loop over circuit lengths m
        for j in range(number_samples): # loop over number of samples N*K
            for k in range(2**number_qubits): # loop over all possible outcomes
                counts[i, j, k] = results[i][j].get_counts().int_outcomes().get(k)

    probs = counts/number_shots
    
    return probs


def draw_sequences(number_qubits, protocol_type, lengths, number_samples):
    """Draw sequences of random Cliffords. Inputs:
    number_qubits
    protocol_type: can be "multi" or "local", depending on the type of UIRS protocol
    lengths: list of circuit lengths
    number_samples: number of sampled circuits for each length 
    """

    sequences = []
    
    if protocol_type == 'multi':
        for length in lengths:
            # seq_list will be a list with all sequences for a given length:
            seq_list = []
            for rep in range(number_samples):
                seq = [qi.random_clifford(number_qubits) for i in range(length)]
                seq_list.append(seq)
            
            # sequences is a list of lists (for different lengths & number of samples per length):
            sequences.append(seq_list)
                
        
    # for now only for two qubits:
    if protocol_type == 'local':
        if number_qubits != 2:
            print("Watch out! Function not yet generalized to more than 2 qubits")
        
        for length in lengths:
            # seq_list will be a list with all sequences for a given length:
            seq_list = []
            for rep in range(number_samples):
                if number_qubits == 2:
                    seq = [qi.random_clifford(1).tensor(qi.random_clifford(1)) for i in range(length)]
                elif number_qubits == 1:
                    seq = [qi.random_clifford(1) for i in range(length)]
                seq_list.append(seq)
            
            # sequences is a list of lists (for different lengths & number of samples per length):
            sequences.append(seq_list)
        
    return sequences


def construct_circuits(sequences, number_qubits):
    """Construct QuantumCircuit objects from the drawn random sequences
    """

    circuits = []
    
    for length in range(len(sequences)):
        circ_list = []
        
        for rep in range(len(sequences[length])):
            circ = QuantumCircuit(number_qubits, number_qubits)
            for gate in range(len(sequences[length][rep])):
                circ.append(sequences[length][rep][gate].to_instruction(), [0,1])
                circ.barrier(range(number_qubits))
            # remove final barrier:
            circ.data.pop()
            # insert measurements at the end:
            circ.measure(range(number_qubits), range(number_qubits))
            circ_list.append(circ)
            
        circuits.append(circ_list)     
    
    return circuits


def get_ptms(sequences):
    """Transform input sequences into PTMs, using Qiskit's PTM function"""

    ptms = []
    
    for length in range(len(sequences)):
        ptm_list = []
        for reps in range(len(sequences[length])):
            gate_list = []
            for gate in range(len(sequences[length][reps])):
                ptm = np.real(qi.PTM(sequences[length][reps][gate]).data)
                gate_list.append(ptm)
                
            ptm_list.append(gate_list)
            
        ptms.append(ptm_list)
    
    return ptms


def get_projection_matrices(protocol_type):
    """Returns the projection matrices P_w for the global and local Clifford UIRS protocol.
    Currently only for 2 qubits. The projection matrices are ordered by their bitstrings w: 00, 01, 10, 11
    """

    if protocol_type == 'local':
        proj00 = np.zeros((16,16))
        proj01 = np.zeros((16,16))
        proj10 = np.zeros((16,16))
        proj11 = np.zeros((16,16))

        indices00 = [0]
        indices01 = [1,2,3]
        indices10 = [4,8,12]
        indices11 = [5,6,7,9,10,11,13,14,15]

        proj00[0, 0] = 1

        for index in indices00:
            proj00[index, index] = 1

        for index in indices01:
            proj01[index, index] = 1
            
        for index in indices10:
            proj10[index, index] = 1
            
        for index in indices11:
            proj11[index, index] = 1

        projection_matrices = [proj00, proj01, proj10, proj11]

    elif protocol_type == 'multi':
        projection_matrices = np.identity(16)
        projection_matrices[0, 0] = 0

    else:
        raise ValueError("Error: protocol type not supported")
        

    return projection_matrices


def get_ptm_subblocks(ptm):
    """Returns the following subblocks of a 16x16 input PTM: 
    01, 10, tensor product(01, 10), 11
    """

    indices01 = [1,2,3]
    indices10 = [4,8,12]
    indices11 = [5,6,7,9,10,11,13,14,15]

    sub01 = np.empty((3,3))
    sub10 = np.empty((3,3))
    for row in range(3):
        for col in range(3):
            sub01[row, col] = ptm[indices01[row], indices01[col]]
            sub10[row, col] = ptm[indices10[row], indices10[col]]
            
    sub11 = np.empty((9,9))
    for row in range(9):
        for col in range(9):
            sub11[row, col] = ptm[indices11[row], indices11[col]]

    tensorprod = np.kron(sub10, sub01)

    return sub01, sub10, tensorprod, sub11


def array_to_sparse(ptms):
    """Transform input PTMs (as numpy arrays) into sparse matrices
    """

    sparse_ptms = []
    
    for length in range(len(ptms)):
        ptm_list = []
        
        for reps in range(len(ptms[length])):
            gate_list = []
            
            for gate in range(len(ptms[length][reps])):
                ptm = sparse.csr_matrix(ptms[length][reps][gate])
                gate_list.append(ptm)
                
            ptm_list.append(gate_list)
            
        sparse_ptms.append(ptm_list)
                
    return sparse_ptms


def sparse_to_array(sparse_ptms):
    """Transform input PTMs saved as sparse matrices into numpy arrays
    """

    ptms = []
    
    for length in range(len(sparse_ptms)):
        ptm_list = []
        
        for reps in range(len(sparse_ptms[length])):
            gate_list = []
            
            for gate in range(len(sparse_ptms[length][reps])):
                ptm = sparse_ptms[length][reps][gate].toarray()
                gate_list.append(ptm)
                
            ptm_list.append(gate_list)
            
        ptms.append(ptm_list)
                
    return ptms


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data


def intersperse(lst, item):
    """Intersperse an item between each elements of a list.
    From https://stackoverflow.com/a/5921708
    """

    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def get_initial_state_vec():
    """Function for building the vectorized initial state, for now only with two qubits
    """

    nQ = 2  # needs to be generalized to arbitrary nQ later
    
    # normalized two-qubit Paulis
    I = 1/np.sqrt(2)*np.array([[1,0],  [0,1]])
    X = 1/np.sqrt(2)*np.array([[0,1],  [1,0]])
    Y = 1/np.sqrt(2)*np.array([[0,-1j], [1j,0]])
    Z = 1/np.sqrt(2)*np.array([[1,0],  [0,-1]])
    
    # build normalized two-qubit Paulis
    paulis = [I, X, Y, Z]
    two_qubit_paulis = [np.kron(i, j) for i in paulis for j in paulis]

    # define initial state rho (all zero state) in the standard basis:
    rho = np.zeros((2**nQ, 2**nQ))
    rho[0, 0] = 1
    
    # initialize and fill vectorized rho
    rho_vec = np.empty(((2**nQ)**2, 1))
    for index in range(len(two_qubit_paulis)):
        
        # the entries are real by definition, so I enforce this explicitly:
        rho_vec[index,0] = np.real(np.trace(two_qubit_paulis[index].conj().T @ rho))

    return rho_vec


def get_two_qubit_cliffords():
    """Returns a list of all 11,520 two-qubit Clifford gates in the standard basis
    """

    I = np.array([[1,0],  [0,1]])
    X = np.array([[0,1],  [1,0]])
    Y = np.array([[0,-1j], [1j,0]])
    Z = np.array([[1,0],  [0,-1]])
    H = 1/np.sqrt(2) * np.array([[1,1],  [1,-1]])
    S = np.array([[1,0],  [0,1j]])

    V = H @ S @ H @ S
    W = V @ V

    # define the sets from which operators are drawn:
    h_set = [I, H]
    v_set = [I, V, W]
    p_set = [I, X, Y, Z]

    # we also need the two possible CNOTs on two qubits:
    CNOT01 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
    CNOT10 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]])

    two_qubit_cliffords = []
    for h0 in h_set:
        for h1 in h_set:
            for v0 in v_set:
                for v1 in v_set:
                    for p0 in p_set:
                        for p1 in p_set:
                            # classes 1 and 4 with 576 elements each:
                            class1 = np.kron(h0, h1) @ np.kron(v0, v1) @ np.kron(p0, p1)
                            class4 = np.kron(h0, h1) @ np.kron(v0, v1) @ CNOT01 @ CNOT10 @ CNOT01 @ np.kron(p0, p1)
                            two_qubit_cliffords.append(class1)
                            two_qubit_cliffords.append(class4)
                            
                            for v0p in v_set:
                                for v1p in v_set:
                                    # classes 2 and 3 with 5184 elements each:
                                    class2 = np.kron(h0, h1) @ np.kron(v0, v1) @ CNOT01 @ np.kron(v0p, v1p) @ np.kron(p0, p1)
                                    class3 = np.kron(h0, h1) @ np.kron(v0, v1) @ CNOT01 @ CNOT10 @ np.kron(v0p, v1p) @ np.kron(p0, p1)
                                    two_qubit_cliffords.append(class2)
                                    two_qubit_cliffords.append(class3)

    return two_qubit_cliffords


def get_measurement_vecs():
    """Function for building the vectorized POVM elements, only for two qubits for now
    """

    nQ = 2
    
    # normalized two-qubit Paulis
    I = 1/np.sqrt(2)*np.array([[1,0],  [0,1]])
    X = 1/np.sqrt(2)*np.array([[0,1],  [1,0]])
    Y = 1/np.sqrt(2)*np.array([[0,-1j], [1j,0]])
    Z = 1/np.sqrt(2)*np.array([[1,0],  [0,-1]])
    
    # build two-qubit Paulis
    paulis = [I, X, Y, Z]
    two_qubit_paulis = [np.kron(i, j) for i in paulis for j in paulis]
    
    povm_list = []
    for i in range(2**nQ):
        povm = np.zeros((2**nQ, 2**nQ))
        povm[i, i] = 1
        povm_list.append(povm)
        
    povm_vecs_list = []
    for i in range(len(povm_list)):
        povm_vec = np.empty((1, (2**nQ)**2))
        
        for index in range(len(two_qubit_paulis)):
                povm_vec[0, index] = np.real(np.trace(povm_list[i] @ two_qubit_paulis[index]))             
            
        povm_vecs_list.append(povm_vec)
    
    return povm_vecs_list


def get_correlators(ptms, probs, A, number_qubits, number_samples, N, K, lengths, protocol_type, projection_matrix, initial_state=None, measurements=None, ci=True):
    """Calculates the correlators k(m) including bootstrapped confidence intervals
    """
    
    correlation_functions = sequence_correlation_function(ptms, A, number_qubits, number_samples, lengths, protocol_type, projection_matrix, initial_state=initial_state, measurements=measurements)

    correlator_array = np.empty((len(lengths), number_samples))
    for i in range(len(lengths)):  # loop over circuit lengths m 
        for j in range(number_samples):  # loop over number of sequence samples N*K
            correlator_array[i, j] = np.nansum(correlation_functions[i, j, :] * probs[i, j, :])

    # calculate the values of k(m) via median-of-means estimation from the samples
    k_values = np.empty(len(lengths))

    if ci is True:
        k_confidence_intervals = np.empty((len(lengths), 2))
    
    for m in range(len(lengths)):
        k_hat = median_of_means(correlator_array[m, :] , K, N)
        k_values[m] = k_hat

        if ci is True:
            confidence_interval = bootstrap(correlator_array[m, :], func=median_of_means, confidence_level=0.95, K=K, N=N)
            # bootstrap the standard deviation:
            #standard_deviation = bootstrap_std(correlator_array[m, :], func=median_of_means, K=K, N=N)
            k_confidence_intervals[m, :] = confidence_interval

    if ci is True:
        return k_values, k_confidence_intervals
    else:
        return k_values


def get_decay_rate(k_values, lengths):
    """Convenience function for calculating only the exponential decay rate of an input list of k(m) values
    To do: switch to lmfit package
    """

    def func(x, a, b):
        return a * b**(x-1)
    
    # if the values immediately drop to zero, set the decay rate zero to avoid errors:
    if np.abs(k_values[1] / k_values[0]) < 1e-5 and np.abs(k_values[2] / k_values[0] < 1e-5):  # checking the second and third value, in case one of them is an outlier
    
        decay_rate = 0

    else:
     
        popt, pcov = curve_fit(func, lengths, k_values)
        decay_rate = popt[1]

    return decay_rate


def median_of_means(samples, K, N):
    """Takes a list of samples and performs the median-of-means algorithm. Inputs:
    samples: list or one-dimensional array
    K: number of independent arithmetic means (chunks)
    N: number of samples used for the calculation of each arithmetic mean
    """
    
    if len(samples) != N*K:
        raise ValueError("Error: number of samples is not N*K")
    
    means = np.empty(K, dtype=np.complex128)
    
    # calculate the mean for each chunk:
    for i in range(K):
        mean = np.mean(samples[i*N:i*N+N])
        means[i] = mean
    
    median = np.median(means)
    
    return median


def bootstrap(data, func, confidence_level=0.95, number_resamples=1000, **kwargs):
    """Returns the bootstrapped confidence interval of a function "func" on a given dataset "data" 
    """

    sample_size = len(data)
    resampled_results = np.empty(number_resamples)
    
    for i in range(number_resamples):
        resample = np.random.choice(data, size=sample_size, replace=True)
        resampled_results[i] = func(resample, **kwargs)
    
    alpha = (1 - confidence_level) / 2 * 100  # in percent
    confidence_interval = np.percentile(resampled_results, [alpha, 100-alpha])
    
    return np.array(confidence_interval)


def bootstrap_std(data, func, number_resamples=1000, **kwargs):
    """Returns the bootstrapped standard deviation of a function "func" on a given dataset "data"
    """

    sample_size = len(data)
    resampled_results = np.empty(number_resamples)
    
    for i in range(number_resamples):
        resample = np.random.choice(data, size=sample_size, replace=True)
        resampled_results[i] = func(resample, **kwargs)

    return np.std(resampled_results)


def custom_unitary(circ, operator, name):
    """(Possibly unneccessary) workaround to create a custom unitary from an input operator
    that can be appended to a circuit "circ"
    """
    
    circ.unitary(operator, [0,1], name) # append custom unitary given by Qiskit operator
    inst = circ.data[-1] # get the data element of the custom unitary
    circ.data.pop() # remove it again
    
    return inst


def insert_crosstalk(circ, errorgate):
    """Inserts an input errorgate after each occurence of an Rx gate on qubit 0
    """
    
    # location of the first qubit's Rx gates:
    xgates = [i for i, j in enumerate(circ.data) if j[0].name == 'rx' and j[1][0].index == 0]
    
    # create a copy of the input circuit (so that the input circuit remains unchanged)
    new_circ = circ.copy()
    
    # after each occurence of an Rx gate on the first qubit, insert the custom errorgate:
    for time, index in enumerate(xgates):
        new_circ.data.insert(index + 1 * (time+1), errorgate)  # the index needs to be adjusted since adding a gate shifts all subsequent indices
    
    return new_circ


def insert_error(circ, errorgate):
    """Inserts an errorgate after every gate in the input circuit
    """
    
    # location of barriers:
    barriers = [i for i, j in enumerate(circ.data) if j[0].name == 'barrier']
    
    # create a copy of the input circuit (so that the input circuit remains unchanged)
    new_circ = circ.copy()
    
    # insert the last error before measurements:
    new_circ.data.insert(len(new_circ.data)-2, errorgate)
    
    # for sequence length m=1, that's it:
    if len(barriers) == 0:
        return new_circ
    
    # now insert it in front of each barrier
    for index in reversed(barriers): # starting at the last barrier so that the insertions don't affect the other indices
        new_circ.data.insert(index, errorgate)
    
    return new_circ


def insert_error_randomized(circ, coherent_error_func, range1, range2):
    """Inserts an errorgate after every gate in the input circuit
    """
    
    # location of barriers:
    barriers = [i for i, j in enumerate(circ.data) if j[0].name == 'barrier']
    
    # create a copy of the input circuit (so that the input circuit remains unchanged)
    new_circ = circ.copy()
    
    # insert the last error before measurements:
    errorgate = custom_unitary(new_circ, qi.Operator(coherent_error_func(range1, range2)), "ZZ")
    new_circ.data.insert(len(new_circ.data)-2, errorgate)
    
    # for sequence length m=1, that's it:
    if len(barriers) == 0:
        return new_circ
    
    # now insert it in front of each barrier
    for index in reversed(barriers): # starting at the last barrier so that the insertions don't affect the other indices
        errorgate = custom_unitary(new_circ, qi.Operator(coherent_error_func(range1, range2)), "ZZ")
        new_circ.data.insert(index, errorgate)
    
    return new_circ


def get_sweep_plot(coherent_error_func, params1, params2, ptms, noisy_data, number_qubits, number_samples, N, K, lengths, params_to_plot, save=False):
    """Calculates an array of generalized fidelities for unitary noise optimization, plots a heatmap and 
    individual curves k(m), and saves the corresponding data, given the inputs:
        coherent_error_func: ansatz for the unitary noise
        params1, params2: parameter values to calculate
        params_to_plot: list of index tuples, determining the parameter values for which k(m) will be plotted
    """

    fit_params = []
    fit_cov = []
    y_vals = []
    confidence_intervals = []

    protocol_type = 'multi'
    projection_matrix = get_projection_matrices('multi')
    
    # calculate curves, fit parameters, confidence intervals:
    for theta1 in params1:
        for theta2 in params2:
    
            coherent_error = coherent_error_func(theta1, theta2)

            A = projection_matrix @ np.real(qi.PTM(qi.Operator(coherent_error))) @ projection_matrix
            
            k_values, ci = get_correlators(ptms, noisy_data, A, number_qubits, number_samples, N, K, lengths, protocol_type, projection_matrix, initial_state=None, measurements=None, ci=True)

            def func(x, a, b):
                return a * b**x

            popt, pcov = curve_fit(func, lengths, k_values)

            fit_params.append(popt)
            fit_cov.append(pcov)
            y_vals.append(k_values)
            confidence_intervals.append(ci)
        
    # extract decay rates and reshape to 2D
    decay_rates = np.array(fit_params)[:,1]
    sweep_array = decay_rates.reshape(len(params1),len(params2))

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(sweep_array, cmap='viridis')

    # set ticks (every second angle)
    ax.set_xticks(np.arange(len(params2)))
    ax.set_yticks(np.arange(len(params1)))

    # label ticks
    ax.set_xticklabels(params2, rotation=45)
    ax.set_yticklabels(params1)

    # axis labels
    ax.set_ylabel(r"$\theta_1$")
    ax.set_xlabel(r"$\theta_2$")

    # add values inside the cells
    #for i in range(len(params1)):
     #   for j in range(len(params2)):
     #       text = ax.text(j, i, "{:.2%}".format(sweep_array[i, j]),
     #                      ha="center", va="center", color="black")
    cb = plt.colorbar(im, ax=ax)

    fig.tight_layout()
    if save is True:
        plt.savefig('unitary_noise_sweep.pdf')
        np.save("unitary_noise_sweep_array", sweep_array)
    
    plt.show()
    
    
    # plotting some of the decays:

    # first setting the color scheme:
    sweep_array_projected = (sweep_array - np.min(sweep_array)) / (np.max(sweep_array - np.min(sweep_array)))
    colors = matplotlib.cm.viridis(sweep_array_projected)
    
    # reshape the array of k(m) values:
    k_values = np.array(y_vals).reshape(len(params1), len(params2), len(lengths))
    
    # reshape the confidence intervals:
    confidence_intervals_reshaped = np.array(confidence_intervals).reshape(len(params1), len(params2), len(lengths), 2)
    
    

    # now plot the data:
    for thetas in params_to_plot:
        
        k_errorbars = np.empty(np.shape(confidence_intervals_reshaped[thetas[0]][thetas[1]]))
        for line in range(len(lengths)):
            k_errorbars[line, :] = np.abs(confidence_intervals_reshaped[thetas[0]][thetas[1]][line][:] - k_values[thetas[0]][thetas[1]][line])
        
        plt.errorbar(lengths, k_values[thetas[0]][thetas[1]], yerr=np.transpose(k_errorbars), xerr=0, 
                     label=r"$\theta_1$ = {:.1f}, $\theta_2$ = {:.1f}".format(params1[thetas[0]], params2[thetas[1]]), 
                     color=colors[thetas[0]][thetas[1]], elinewidth=0.8, solid_capstyle='projecting', capsize=2)

    plt.legend()
    plt.xlabel(r"Sequence length $m$")
    plt.ylabel(r"$k(m)$")
    if save is True:
        plt.savefig('unitary_noise_decays.pdf')
        np.save("unitary_noise_k_values", k_values)
        np.save("unitary_noise_confidence_intervals", confidence_intervals_reshaped)
    plt.show()
    
    return


def nelder_mead_optimization(coherent_error_func, ptms, noisy_data, number_qubits, number_samples, N, K, lengths):

    projection_matrix = get_projection_matrices('multi')

    def optimization_func(params, coherent_error_func, ptms, noisy_data, number_qubits, number_samples, N, K, lengths, projection_matrix, info):
        
        coherent_error = coherent_error_func(params[0], params[1])

        A = projection_matrix @ np.real(qi.PTM(qi.Operator(coherent_error))) @ projection_matrix

        k_values = get_correlators(ptms, noisy_data, A, number_qubits, number_samples, N, K, lengths, 'multi', projection_matrix, initial_state=None, measurements=None, ci=False)

        def func(x, a, b):
            return a * b**x

        popt, pcov = curve_fit(func, lengths, k_values)
            
        # write status of the optimization to a file:
        savefile.write('{0:4d}  {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f} '.format(info['Nfeval'], params[0], params[1], popt[1], np.sqrt(np.diag(pcov))[1]))
        savefile.write("\n")
            
        info['Nfeval'] += 1
            
        return -popt[1] # negative to use the minimization algorithms
    
    opt = {'maxiter':100, 'disp': True}

    savefile = open('nelder_mead_path.txt', 'w')
    res = scipy.optimize.minimize(optimization_func, x0=[0, 0.15], 
                                    args=(coherent_error_func, ptms, noisy_data, number_qubits, number_samples, N, K, lengths, projection_matrix, {'Nfeval':0}), 
                                    method='Nelder-Mead', options=opt)
    savefile.close()

    return


def rearrange_ptm(input_ptm):
    """Rearranges a PTM from Qiskit's standard ordering to the one used in plots
    """
    
    ptm = input_ptm.copy()
    temp = input_ptm.copy()

    ptm[5,:] = temp[8,:]
    ptm[6,:] = temp[12,:]
    ptm[7, :] = temp[5, :]
    ptm[8, :] = temp[6, :]
    ptm[9, :] = temp[7, :]
    ptm[10, :] = temp[9, :]
    ptm[11, :] = temp[10, :]
    ptm[12, :] = temp[11, :]
    
    temp = ptm.copy()
    
    ptm[:, 5] = temp[:, 8]
    ptm[:, 6] = temp[:, 12]
    ptm[:, 7] = temp[:, 5]
    ptm[:, 8] = temp[:, 6]
    ptm[:, 9] = temp[:, 7]
    ptm[:, 10] = temp[:, 9]
    ptm[:, 11] = temp[:, 10]
    ptm[:, 12] = temp[:, 11]

    return ptm


def plot_ptm(ptm, save=False, filename=None):
    """Rearranges and plots a PTM
    """

    fig, ax = plt.subplots()

    im = ax.imshow(rearrange_ptm(ptm), cmap="RdBu", vmin=-1, vmax=1)
    labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    cb = plt.colorbar(im, ax=ax, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    if save is True:
        plt.savefig(filename)
    plt.show()

    return


def plot_diff(ptm, save=False, filename=None):
    """Plots the "crosstalk characterization matrix" Lambda_11 - tensorproduct(Lambda_10, Lambda_01)
    """

    sub01, sub10, tensorprod, sub11 = get_ptm_subblocks(ptm)
    
    fig, ax = plt.subplots()

    im = ax.imshow(sub11 - tensorprod, cmap="RdBu", vmin=-0.5, vmax=0.5)
    labels = [''.join(x) for x in itertools.product('XYZ', repeat=2)]
    cb = plt.colorbar(im, ax=ax, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    if save is True:
        plt.savefig(filename)
    plt.show()
    
    return