import numpy as np
import scipy
import scipy.linalg as sg

def generalized_eigenvalue_covariance_mean(covariance_matrices, 
                                           tolerance=1e-9, 
                                           max_iterations=50,
                                           norm_type='frobenius',
                                           initialization=None):
    """
    Computes the generalized eigenvalue mean of a set of covariance matrices.

    Parameters:
    covariance_matrices (np.ndarray): A 3D array of shape (n_matrices, matrix_size, matrix_size) containing the covariance matrices.
    tolerance (float): The tolerance for the stopping criterion. Default is 1e-9.
    max_iterations (int): The maximum number of iterations. Default is 50.
    norm_type (str): The type of norm used for the stopping criterion. Options are 'frobenius' or None. Default is 'frobenius'.
    initialization (str or np.ndarray): The initial value for the mean covariance matrix. Options are 'mean', 'identity', 'first', or a specific matrix.
                                         If None, the mean of the covariance matrices is used. Default is None.
    
    Returns:
    np.ndarray: The mean covariance matrix.
    """
    num_matrices, matrix_size, _ = covariance_matrices.shape

    # Define initial value for the mean covariance matrix
    # Define initial value for the mean covariance matrix
    if initialization == 'mean' or initialization is None:
        mean_covariance = np.mean(covariance_matrices, axis=0)
    elif initialization == 'identity':
        mean_covariance = np.identity(matrix_size)
    elif initialization == 'first':
        mean_covariance = covariance_matrices[0]
    else:
        mean_covariance = initialization

    iteration = 0
    criterion = np.finfo(np.float64).max

    # Iterative algorithm to compute the mean covariance matrix
    while (criterion > tolerance) and (iteration < max_iterations):
        iteration += 1
        J_matrix = np.zeros((matrix_size, matrix_size))

        for cov_matrix in covariance_matrices:
            eigenvalues, eigenvectors = scipy.linalg.eigh(cov_matrix, mean_covariance)
            log_eigenvalues = np.diag(np.log(eigenvalues))
            J_matrix += eigenvectors @ log_eigenvalues @ eigenvectors.T

        J_matrix = (1 / num_matrices) * mean_covariance @ J_matrix @ mean_covariance
        
        # Compute the norm
        if norm_type is None:
            eigenvalues, _ = scipy.linalg.eigh(J_matrix, check_finite=False)
            criterion = np.sqrt(np.sum(eigenvalues ** 2))
        elif norm_type == 'frobenius':
            criterion = np.linalg.norm(J_matrix, ord='fro')

        # Update the mean covariance matrix
        eigenvalues, eigenvectors = scipy.linalg.eigh(J_matrix, mean_covariance)
        exp_eigenvalues = np.diag(np.exp(eigenvalues))
        mean_covariance = mean_covariance @ eigenvectors @ exp_eigenvalues @ eigenvectors.T @ mean_covariance
        mean_covariance = 0.5 * (mean_covariance + mean_covariance.T)

    return np.array(mean_covariance)
