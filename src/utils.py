import scipy
import scipy.linalg as sg
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from src.covariance_means import generalized_eigenvalue_covariance_mean

def regularize_covariance_matrices(covariance_matrices):
    """
    Regularizes a list of covariance matrices by ensuring all eigenvalues are positive.
    If any eigenvalue is non-positive, a small value (1e-9) is added to all eigenvalues.

    Parameters:
    covariance_matrices (list of np.ndarray): A list of covariance matrices to be regularized.

    Returns:
    np.ndarray: A list of regularized covariance matrices.
    """
    size = covariance_matrices[0].shape[0]
    regularized_matrices = []
    min_eigenvalues = []

    # Find the minimum eigenvalue across all covariance matrices
    for cov_matrix in covariance_matrices:
        eigenvalues = sg.eigh(cov_matrix, check_finite=False, eigvals_only=True)
        min_eigenvalues.append(min(eigenvalues))

    global_min_eigenvalue = min(min_eigenvalues)

    # Regularize the matrices
    if global_min_eigenvalue <= 0:
        epsilon = np.abs(global_min_eigenvalue) + 1e-9
        for cov_matrix in covariance_matrices:
            regularized_matrix = cov_matrix + epsilon * np.eye(size)
            regularized_matrices.append(regularized_matrix)
    else:
        epsilon = 1e-9
        for cov_matrix in covariance_matrices:
            regularized_matrix = cov_matrix + epsilon * np.eye(size)
            regularized_matrices.append(regularized_matrix)

    return np.array(regularized_matrices)

def load_and_regularize_covariance_matrices(deep_features_type='covs_DN20', group_type='CG', num_patients=11, num_videos=8):
    """
    Loads and regularizes covariance matrices for a given dataset.

    Parameters:
    deep_features_type (str): The type of deep features.
    group_type (str): The group type.
    num_patients (int): The number of patients.
    num_videos (int): The number of videos per patient.

    Returns:
    list: A list containing regularized covariance matrices for each patient and video.
          The structure is (n_patients, n_videos, n_frames, N, N).
    """
    dataset = []  # (n_patients, n_videos, n_frames, N, N)
    
    for patient_id in tqdm(range(1, num_patients + 1), desc='Patients'):
        patient_videos_cov_matrices = []  # (n_videos, n_frames, N, N)
        
        for video_id in range(1, num_videos + 1):           
            cov_matrices_per_frame = np.load(f'data/{deep_features_type}/{group_type}_{patient_id:02d}_{video_id:02d}.npy', allow_pickle=True)  # (n_frames, N, N)
            cov_matrices_per_frame = regularize_covariance_matrices(cov_matrices_per_frame)            
            patient_videos_cov_matrices.append(cov_matrices_per_frame)
        
        dataset.append(patient_videos_cov_matrices)
    
    return dataset


def calculate_covariance_means(dataset, tolerance=1e-9, max_iterations=50, initialization=None, norm_type='frobenius'):
    """
    Calculates the covariance mean for each patient's video in the dataset.

    Parameters:
    dataset (list): A nested list structure containing covariance matrices for each patient and video.
                    The structure is (n_patients, n_videos, n_frames, N, N).
    tolerance (float): The tolerance for the stopping criterion in the mean calculation. Default is 1e-9.
    max_iterations (int): The maximum number of iterations for the mean calculation. Default is 50.
    initialization (str or np.ndarray): The initial value for the mean covariance matrix. Options are 'mean', 'identity', 'first', or a specific matrix.
                                         If None, the mean of the covariance matrices is used. Default is None.
    norm_type (str): The type of norm used for the stopping criterion. Options are 'frobenius' or None. Default is 'frobenius'.

    Returns:
    list: A list containing the covariance mean matrices for each patient's video.
    """
    covariance_means = []

    for patient_data in tqdm(dataset, desc='Patients [Covariance mean]'):
        patient_cov_means = []

        for video_data in patient_data:
            mean_cov_matrix = generalized_eigenvalue_covariance_mean(
                np.array(video_data),
                tolerance=tolerance,
                max_iterations=max_iterations,
                initialization=initialization,
                norm_type=norm_type
            )
            patient_cov_means.append(mean_cov_matrix)

        covariance_means.append(patient_cov_means)

    return covariance_means

def matrix_operator(matrix, operator):
    """
    Applies an element-wise operator to the eigenvalues of a matrix.

    Parameters:
    matrix (np.ndarray): The input matrix.
    operator (function): The operator to apply to the eigenvalues.

    Returns:
    np.ndarray: The resulting matrix after applying the operator to its eigenvalues.
    """
    eigvals, eigvects = sg.eigh(matrix, check_finite=False)
    eigvals_transformed = np.diag(operator(eigvals))
    result = np.dot(np.dot(eigvects, eigvals_transformed), eigvects.T)
    return result

def sqrtm(matrix):
    """
    Computes the matrix square root.

    Parameters:
    matrix (np.ndarray): The input matrix.

    Returns:
    np.ndarray: The square root of the input matrix.
    """
    return matrix_operator(matrix, np.sqrt)

def invsqrtm(matrix):
    """
    Computes the inverse matrix square root.

    Parameters:
    matrix (np.ndarray): The input matrix.

    Returns:
    np.ndarray: The inverse square root of the input matrix.
    """
    return matrix_operator(matrix, lambda x: 1.0 / np.sqrt(x))

def log_AB(A, B):
    """
    Computes the matrix logarithm of A in the basis of B.

    Parameters:
    A (np.ndarray): The input matrix A.
    B (np.ndarray): The input matrix B.

    Returns:
    np.ndarray: The logarithm of matrix A in the basis of matrix B.
    """
    B_sqrt = sqrtm(B)
    B_invsqrt = invsqrtm(B)

    L = np.dot(np.dot(B_invsqrt, A), B_invsqrt)
    L_log = sg.logm(L)

    result = np.dot(np.dot(B_sqrt, L_log), B_sqrt)
    return result


def project_to_tangent_and_triu(covariance_means):
    """
    Projects each covariance matrix in the dataset to the tangent space and extracts the upper triangular part.

    Parameters:
    covariance_means (list): A list containing mean covariance matrices for each patient's video.
                             The structure is (n_patients, n_videos, N, N).

    Returns:
    list: A list of vectors representing the upper triangular part of the log tangent space matrices.
          The structure is (n_patients, n_videos, vector_length).
    """
    matrix_size, _ = covariance_means[0][0].shape
    reff = np.identity(matrix_size)
    
    vectors = []

    for patient_data in tqdm(covariance_means, desc='Patients [Tangent space]'):
        vectors_patient = []

        for video_data in patient_data:
            cov_matrix = video_data
            log_cov_matrix = log_AB(cov_matrix, reff)

            upper_triangular_vector = []
            for k in range(matrix_size):
                for l in range(k, matrix_size):
                    upper_triangular_vector.append(log_cov_matrix[k, l])

            vectors_patient.append(np.array(upper_triangular_vector))

        vectors.append(np.array(vectors_patient))

    return np.array(vectors)


# def metrics_manyTimes(dataset_vectors,
#                       dataset_labels,
#                       veces,
#                       est=None):
#     # X shape: (22, 8, 210)
#     # y shape: (22, )
#     num_patients, num_videos, _ = dataset_vectors.shape
#     many_accuracies=[]
#     many_precisions=[]
#     many_recall=[]
#     many_f1score=[]
#     for i in range(veces):        
#         loo = LeaveOneOut()        
#         for train_index, test_index in loo.split(dataset_vectors): # X= [P1,P2,...,P22] donde Pi=[U1,U2,...,U8]
#             print( train_index, test_index )            
#             X_test = dataset_vectors[test_index[0]] 
#             y_test = dataset_labels[test_index[0]] # scalar
#             y_test = np.array([y_test]*num_videos) # vector
                            
#             X_train=[]
#             y_train=[]
#             for ind in train_index:
#                 X_train += list(dataset_vectors[ind]) #Se arma X_train para que quede con (22-1)*8=168 descriptores    
#                 y_train += list(dataset_vectors[ind])

#             X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
            
#             print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            
#             est.fit(X_train,y_train)
#             y_pred = est.predict(X_test)
                
#             #Se guardan todas las metricas en una respectiva super lista que contendra: veces*22 valores
#             many_accuracies.append(accuracy_score(y_test,y_pred))
#             many_precisions.append(precision_score(y_test, y_pred, average='macro' , zero_division=0))    
#             many_recall.append(recall_score(y_test, y_pred, average='macro' , zero_division=0))
#             many_f1score.append(f1_score(y_test, y_pred, average='macro' , zero_division=0))
   

#     #Las metricas finales seran la media y la std de la respectiva super lista
#     display(pd.DataFrame({'Accuracy': [np.mean(many_accuracies)], 
#                           'std accuracy': [np.std(many_accuracies)],
#                           'Precision': [np.mean(many_precisions)], 
#                           'std precision': [np.std(many_precisions)],
#                           'Recall': [np.mean(many_recall)], 
#                           'std recall': [np.std(many_recall)],
#                           'F1Score': [np.mean(many_f1score)], 
#                           'std F1score': [np.std(many_f1score)],
#                          })) 
#     return

def evaluate_metrics_repeatedly(dataset_vectors, dataset_labels, repetitions = 1, estimator=None):
    """
    Evaluates classification metrics repeatedly using Leave-One-Out cross-validation.

    Parameters:
    dataset_vectors (np.ndarray): The input dataset vectors of shape (n_patients, n_videos, n_features).
    dataset_labels (np.ndarray): The labels for the dataset of shape (n_patients, ).
    repetitions (int): The number of repetitions for the evaluation.
    estimator (object): The machine learning estimator to use for fitting and prediction.

    Returns:
    None
    """
    num_patients, num_videos, _ = dataset_vectors.shape
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for _ in tqdm(range(repetitions), desc='Repetitions'):
        loo = LeaveOneOut()

        for train_index, test_index in loo.split(dataset_vectors):
            X_test = dataset_vectors[test_index[0]]
            y_test = np.array([dataset_labels[test_index[0]]] * num_videos)
            
            X_train = []
            y_train = []
            for ind in train_index:
                X_train.extend(dataset_vectors[ind])
                y_train.extend([dataset_labels[ind]] * num_videos)

            X_train, X_test = np.array(X_train), np.array(X_test)
            y_train, y_test = np.array(y_train), np.array(y_test)
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
                
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    metrics_summary = pd.DataFrame({
        'Accuracy': [np.mean(accuracies)], 
        'std Accuracy': [np.std(accuracies)],
        'Precision': [np.mean(precisions)], 
        'std Precision': [np.std(precisions)],
        'Recall': [np.mean(recalls)], 
        'std Recall': [np.std(recalls)],
        'F1 Score': [np.mean(f1_scores)], 
        'std F1 Score': [np.std(f1_scores)],
    })
    
    display(metrics_summary)
    
def reduce_spd_dataset(dataset_covs, n_components):
    dataset_covs = np.array(dataset_covs)
    new_dataset = np.zeros((dataset_covs.shape[0], dataset_covs.shape[1], n_components, n_components))
    for i in range(dataset_covs.shape[0]):
        for j in range(dataset_covs.shape[1]):
            covariance_to_reduce = dataset_covs[i, j]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_to_reduce) # Calculate eigenvalues and eigenvectors            
            idx = eigenvalues.argsort()[::-1] # Sort eigenvalues in descending order
            eigenvectors = eigenvectors[:,idx][:,:n_components] # Sort eigenvectors according to eigenvalues and get the first n_components
            reduced_covariance = eigenvectors.T @ np.diag(eigenvalues) @ eigenvectors            
            new_dataset[i, j] = reduced_covariance            
    return new_dataset    

