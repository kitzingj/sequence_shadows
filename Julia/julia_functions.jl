using PyCall
using LinearAlgebra
using BenchmarkTools
using LsqFit
using Statistics
using NPZ
using StatsBase


function sequence_correlation_function(ptms_converted, A, number_qubits, number_samples, lengths, protocol_type, projection_matrix, initial_state, measurements)
    
    # normalization factors depending on protocol type:
    if protocol_type == "multi"
        alpha = 2^number_qubits + 1;
        
    elseif protocol_type == "local00"
        alpha = 2^number_qubits;
        
    elseif protocol_type == "local01" || protocol_type == "local10"
        alpha = 2^number_qubits * 3;
        
    elseif protocol_type == "local11"
        alpha = 2^number_qubits * 3^2;
    end
    
    # get the PTM for B_x (initial state and measurement)
    B_list = Array{Matrix{Float64}}(undef, 2^number_qubits);
    for povm in 1:4
        B_list[povm] = projection_matrix * initial_state * measurements[povm];
    end
                
    
    correlation_functions = zeros(size(lengths)[1], number_samples, 2^number_qubits);
    gates_matrix = zeros(16, 16);
    temp1 = zeros(16, 16);
    
    @inbounds for i in 1:size(lengths)[1] 
        @inbounds for j in 1:number_samples
            @inbounds for gate in 1:size(ptms_converted[i, j])[1]
                if gate == 1
                    mul!(gates_matrix, projection_matrix, I)
                    mul!(temp1, ptms_converted[i, j][gate], gates_matrix)
                    mul!(gates_matrix, projection_matrix, temp1)

                else
                    mul!(temp1, A, gates_matrix)
                    mul!(gates_matrix, projection_matrix, temp1)
                    mul!(temp1, ptms_converted[i, j][gate], gates_matrix)
                    mul!(gates_matrix, projection_matrix, temp1)
                end
            end

            @inbounds for k in 1:2^number_qubits # loop over all possible outcomes
                # calculate the sequence correlation function (Eq. 12)
                mul!(temp1, B_list[k], gates_matrix)
                correlation_functions[i, j, k] = alpha * tr(temp1);
            end
        end
    
    end
    
    return correlation_functions;
end;


function sequence_correlation_function_views(ptms_converted, A, number_qubits, number_samples, lengths, protocol_type, initial_state, measurements)

    # normalization factors depending on protocol type:
    if protocol_type == "multi"
        alpha = 2^number_qubits + 1;
        
    elseif protocol_type == "local00"
        alpha = 2^number_qubits;
        indices = [1];
        
    elseif protocol_type == "local01"
        alpha = 2^number_qubits * 3;
        indices = [2,3,4];
        
    elseif protocol_type == "local10"
        alpha = 2^number_qubits * 3;
        indices = [5,9,13];
        
    elseif protocol_type == "local11"
        alpha = 2^number_qubits * 3^2;
        indices = [6,7,8,10,11,12,14,15,16];
    end
    
    # get the PTM for B_x (initial state and measurement)
    B_list = Array{Matrix{Float64}}(undef, 2^number_qubits);
    for povm in 1:4
        B_list[povm] = initial_state * measurements[povm];
    end
                
    
    correlation_functions = zeros(size(lengths)[1], number_samples, 2^number_qubits);
    gates_matrix = zeros(16, 16);
    temp1 = zeros(16, 16);
    
    @views begin 
        @inbounds for i in 1:size(lengths)[1] 
            @inbounds for j in 1:number_samples
                @inbounds for gate in 1:size(ptms_converted[i, j])[1]
                    if gate == 1
                        mul!(gates_matrix[indices, indices], ptms_converted[i, j][gate][indices, indices], I)

                    else
                       mul!(temp1[indices, indices], A[indices, indices], gates_matrix[indices, indices])
                       mul!(gates_matrix[indices, indices], ptms_converted[i, j][gate][indices, indices], temp1[indices, indices])
                    end
                end

                @inbounds for k in 1:2^number_qubits # loop over all possible outcomes
                    # calculate the sequence correlation function (Eq. 12)
                    mul!(temp1[indices, indices], B_list[k][indices, indices], gates_matrix[indices, indices])
                    correlation_functions[i, j, k] = alpha * tr(temp1[indices, indices]);
                end
            end
        end
    end
        
    
    return correlation_functions;
end;


function median_of_means(samples, K, N)
    """Takes a list of samples and performs the median-of-means algorithm. Inputs:
    samples: list or one-dimensional array
    K: number of independent arithmetic means (chunks)
    N: number of samples used for the calculation of each arithmetic mean"""
    
    if size(samples)[1] != N*K
        print("Error: number of samples is not N*K")
    end
    
    means = zeros(K)
    
    # calculate the mean for each chunk:
    for i in 0:K-1
        @views means[i+1] = mean(samples[(i*N + 1):(i*N+N)])
    end
    
    median_of_means = median(means)
    
    return median_of_means
end


function median_of_means_bootstrap(data, number_resamples, confidence_level, K, N)
    
    sample_size = size(data)[1]
    resampled_results = zeros(number_resamples)
    
    for i in 1:number_resamples
        resample = sample(data, sample_size, replace=true)
        resampled_results[i] = median_of_means(resample, K, N)
    end
    
    # confidence interval:
    alpha = (1 - confidence_level) / 2
    lower = quantile!(resampled_results, alpha)
    upper = quantile!(resampled_results, 1-alpha)
    
    # variance:
    #variance = var(resampled_results)
    
    return [lower, upper]
end


function get_correlators_views(ptms, probs, A, number_qubits, number_samples, N, K, lengths, protocol_type, initial_state, measurements, ci)
    
    correlation_functions = sequence_correlation_function_views(ptms, A, number_qubits, number_samples, lengths, protocol_type, initial_state, measurements)

    correlator_array = zeros(size(lengths)[1], number_samples);
    
    for j in 1:number_samples
        for i in 1:size(lengths)[1]
            @views correlator_array[i, j] = dot(correlation_functions[i, j, :], probs[i, j, :])
        end
    end

    # calculate the values of k(m) via median-of-means estimation from the samples
    k_values = zeros(size(lengths)[1])
    if ci == true
        k_variance = zeros(size(lengths)[1])
    end

    for m in 1:size(lengths)[1]
        @views k_values[m] = median_of_means(correlator_array[m, :], K, N)
        if ci == true
            @views k_variance[m] = median_of_means_bootstrap(correlator_array[m, :], 1000, 0.95, K, N)
        end
    end
    

    if ci == true
        return k_values, k_variance
    else
        return k_values
    end
                        
end


function get_correlators(ptms, probs, A, number_qubits, number_samples, N, K, lengths, protocol_type, projection_matrix, initial_state, measurements, ci)
    
    correlation_functions = sequence_correlation_function(ptms, A, number_qubits, number_samples, lengths, protocol_type, projection_matrix, initial_state, measurements)

    correlator_array = zeros(size(lengths)[1], number_samples);
    
    for j in 1:number_samples
        for i in 1:size(lengths)[1]
            @views correlator_array[i, j] = dot(correlation_functions[i, j, :], probs[i, j, :])
        end
    end

    # calculate the values of k(m) via median-of-means estimation from the samples
    k_values = zeros(size(lengths)[1])
    if ci == true
        confidence_intervals = zeros(size(lengths)[1], 2)
    end

    for m in 1:size(lengths)[1]
        @views k_values[m] = median_of_means(correlator_array[m, :], K, N)
        if ci == true
            @views confidence_intervals[m, :] = median_of_means_bootstrap(correlator_array[m, :], 1000, 0.95, K, N)
        end
    end
    
    if ci == true
        return k_values, confidence_intervals
    else
        return k_values
    end
                        
end


function ZZ(theta1, theta2)
    
    coherent_error = [[exp(1im/2*(-theta2 - theta1)), 0, 0, 0],
            [0, exp(1im/2*(-theta2 + theta1)), 0, 0],
            [0, 0, exp(1im/2*(theta2 - theta1)), 0],
            [0, 0, 0, exp(1im/2*(theta2 + theta1))]]
    
    return coherent_error
end


function get_sweep_decays(coherent_error_func, params1, params2, ptms, noisy_data, number_qubits, number_samples, N, K, lengths, protocol_type, projection_matrix, initial_state, measurements, ci)

    y_vals = zeros(size(params1)[1], size(params2)[1], size(lengths)[1])
    confidence_intervals = zeros(size(params1)[1], size(params2)[1], size(lengths)[1], 2)
    decay_rates = zeros(size(params1)[1], size(params2)[1])
    
    # calculate curves, fit parameters, confidence intervals:
    for theta2 in 1:size(params2)[1]
        for theta1 in 1:size(params1)[1]
    
            coherent_error = coherent_error_func(params1[theta1], params2[theta2])

            A = projection_matrix * np.real(qi.PTM(qi.Operator(coherent_error))) * projection_matrix
            
            k_values, ci = get_correlators(ptms, noisy_data, A, number_qubits, number_samples, N, K, lengths, protocol_type, projection_matrix, initial_state, measurements, ci)
            
            print("theta1:", params1[theta1])
            print("theta2:", params2[theta2])
            print(k_values)
            print(ci)
            
            decay_rate = get_decay_rate(k_values, lengths)

            y_vals[theta1, theta2, :] = k_values 
            confidence_intervals[theta1, theta2, :, :] = ci
            decay_rates[theta1, theta2] = decay_rate
        end
    end

    return y_vals, confidence_intervals, decay_rates
end


function get_decay_rate_with_variance(k_values, variances, lengths)
# there's still a problem when a variance is zero!
    @. model(x, p) = p[1]*p[2]^(x-1)
    p0 = [0.5, 0.0];
    weights = 1 ./ variances;
    fit = curve_fit(model, lengths, k_values, weights, p0);
    decay_rate = fit.param[2]

    return decay_rate
        
end


function get_decay_rate(k_values, lengths)

    @. model(x, p) = p[1]*p[2]^(x-1)
    p0 = [0.5, 0.0];
    fit = curve_fit(model, lengths, k_values, p0);
    decay_rate = fit.param[2]

    return decay_rate
        
end


function replace_nan(x)
    for i = eachindex(x)
        if isnan(x[i])
            x[i] = 0
        end
    end
end;


# load python functions: (workaround because PyCall's pyimport() was not stable)
py"""
import pickle
import numpy as np
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data

def get_projection_matrices():

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

    return projection_matrices


def get_initial_state_vec():

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
        
        # the entries are again real by definition, so I enforce this explicitly:
        rho_vec[index,0] = np.real(np.trace(two_qubit_paulis[index].conj().T @ rho))

    return rho_vec

def get_two_qubit_cliffords():

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


def get_measurement_vecs(number_qubits=2):

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
"""

load_pickle = py"load_pickle"
get_projection_matrices = py"get_projection_matrices";
get_initial_state_vec = py"get_initial_state_vec";
get_measurement_vecs = py"get_measurement_vecs";
get_two_qubit_cliffords = py"get_two_qubit_cliffords";
