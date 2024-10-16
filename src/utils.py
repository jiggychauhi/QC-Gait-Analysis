import scipy
import scipy.linalg as sg
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pyriemann.utils import mean_covariance
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

def load_and_split_subjects_by_group(pickle_file_path):
    """
    Args:
        pickle_file_path (str): The path to the pickle file containing a dictionary where keys represent subject IDs
                                and values represent covariance matrices.                                
    Returns:
        tuple: A tuple containing four lists:
            - trainval_control_subjects (list): List of control group subjects for training and validation (first 11).
            - trainval_parkinson_subjects (list): List of Parkinson's group subjects for training and validation (first 11).
            - test_control_subjects (list): List of control group subjects for testing (remaining subjects).
            - test_parkinson_subjects (list): List of Parkinson's group subjects for testing (remaining subjects).
    """
    
    with open(pickle_file_path, 'rb') as file:
        subject_covariances = pickle.load(file)
        
    control_subjects = [subject_id for subject_id in subject_covariances.keys() if 'CG' in subject_id]
    parkinson_subjects = [subject_id for subject_id in subject_covariances.keys() if 'PG' in subject_id]
    
    # Split subjects into training/validation (first 11) and testing sets (remaining subjects)
    trainval_control_subjects = control_subjects[:11]
    trainval_parkinson_subjects = parkinson_subjects[:11]
    test_control_subjects = control_subjects[11:]
    test_parkinson_subjects = parkinson_subjects[11:]
        
    return trainval_control_subjects, trainval_parkinson_subjects, test_control_subjects, test_parkinson_subjects, subject_covariances

    

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

def batch_covs(dataset, batch_size, metric):
    """
    Group frames inside a video into batch.
    Compute the mean of this group.

    Parameters:
    dataset (list): A list containing the mean covariance matrices by subject x video x frame                               
    batch_size (int): A list of subject IDs for which the covariance means should be calculated.    
    metric (string): metric to use to compute the mean.

    Returns:
    numpy_array: a list of covariance matrices. 
    groups: Contain the subject number for each covariance matrix.
    """
    n_subject = len(dataset)
    n_video = len(dataset[0])
    covs = []
    groups = []
    for subject in range(n_subject):
        for video in range(n_video):
            n_frames = len(dataset[subject][video])
            for start_batch in range(0, n_frames, batch_size):
                end_betch = min(start_batch + batch_size, n_frames)
                frames = dataset[subject][video][start_batch:end_betch]
                m = mean_covariance(frames, metric=metric)
                covs.append(m)
                groups.append(subject)
    return np.array(covs), groups

def calculate_covariances_per_subject_and_frame(
        subject_covariances,
        ids, 
    ):
    """
    Calculates the mean covariance by subject x video x frame.
    Parameters:
    subject_covariances (dict): A dictionary where keys are subject IDs, and values are dictionaries containing video keys 
                                and their corresponding covariance matrices.
                                Format: {subject_id: {video_key: covariance_matrices}}                                
    ids (list): A list of subject IDs for which the covariance means should be calculated.    

    Returns:
    list: A list containing the mean covariance matrices by subject x video x frame
    """
    
    ret = []
    
    
    for subject_id in ids: # Iterate over each subject in the provided IDs list 
        ret.append([])       
        for video_cov_matrices in subject_covariances[subject_id].values():   # Iterate over each video within the subject's data            
            # video_cov_mean = generalized_eigenvalue_covariance_mean(
            #     covariance_matrices=video_cov_matrices,
            # ) 
            ret[-1].append(video_cov_matrices)
    
    return ret


def calculate_mean_covariances_per_subject(
        subject_covariances,
        ids, 
        tolerance=1e-9, 
        max_iterations=50, 
        initialization=None, 
        norm_type='frobenius'
    ):
    """
    Calculates the mean covariance matrix for each video associated with a given set of subjects.

    Parameters:
    subject_covariances (dict): A dictionary where keys are subject IDs, and values are dictionaries containing video keys 
                                and their corresponding covariance matrices.
                                Format: {subject_id: {video_key: covariance_matrices}}                                
    ids (list): A list of subject IDs for which the covariance means should be calculated.    
    tolerance (float): The tolerance for the stopping criterion in the mean calculation. Default is 1e-9.    
    max_iterations (int): The maximum number of iterations for the mean calculation. Default is 50.    
    initialization (str or np.ndarray): The initial value for the mean covariance matrix. 
                                        Options are 'mean', 'identity', 'first', or a specific matrix. 
                                        If None, the mean of the covariance matrices is used. Default is None.                                        
    norm_type (str): The type of norm used for the stopping criterion. Options are 'frobenius' or None. Default is 'frobenius'.

    Returns:
    dict: A dictionary containing the mean covariance matrices for each video of each subject.
          The keys are a combination of subject IDs and video keys, separated by a hyphen.
          Format: {subject_id-video_key: mean_covariance_matrix}
    """
    
    new_subjects_dict = {}
    
    
    for subject_id in ids: # Iterate over each subject in the provided IDs list        
        for video_key, video_cov_matrices in subject_covariances[subject_id].items():   # Iterate over each video within the subject's data            
            # Calculate the mean covariance matrix for the video's data
            video_cov_mean = generalized_eigenvalue_covariance_mean(
                covariance_matrices=video_cov_matrices,
                tolerance=tolerance,
                max_iterations=max_iterations,
                initialization=initialization,
                norm_type=norm_type
            )            
            # Store the result in the new dictionary, using a combined key of subject ID and video key
            new_subjects_dict[f'{subject_id}-{video_key}'] = video_cov_mean
    
    return new_subjects_dict


def apply_functions_to_dict_arrays(array_dict, function_list):
    """
    Applies a list of functions sequentially to each array in a dictionary.

    Parameters:
    array_dict (dict): A dictionary where each key is a videoID, and each value is a NumPy array of shape (N, N).
    function_list (list): A list of functions to be applied sequentially to each array.

    Returns:
    dict: A new dictionary with the same keys but transformed arrays after applying the functions.
    """
    transformed_dict = {}

    for video_id, array in array_dict.items():
        transformed_array = array
        for func in function_list:
            transformed_array = func(transformed_array)        
        transformed_dict[video_id] = transformed_array
    return transformed_dict


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

def project_to_tangent_and_triu_single(covariance_mean):
    """
    Projects the given covariance matrix to the tangent space and extracts the upper triangular part.

    Parameters:
    covariance_mean (np.ndarray): A mean covariance matrix of shape (N, N).

    Returns:
    np.ndarray: A vector representing the upper triangular part of the log tangent space matrix.
    """
    
    matrix_size = covariance_mean.shape[0]
    reff = np.identity(matrix_size)
    log_cov_matrix = log_AB(covariance_mean, reff)

    # Extract the upper triangular part
    upper_triangular_vector = []
    for k in range(matrix_size):
        for l in range(k, matrix_size):
            upper_triangular_vector.append(log_cov_matrix[k, l])

    return np.array(upper_triangular_vector)

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


def evaluate_subject_based_metrics(dataset_dict, repetitions=1, estimator=None):
    """
    Evaluates classification metrics using Leave-One-Subject-Out cross-validation. Each subject's videos are treated as a group,
    and the model is evaluated by leaving out all samples (videos) from one subject at a time.

    Parameters:
    dataset_dict (dict): A dictionary where the keys are in the format "subjectID-videoID" and the values are feature vectors.
                         The key should also contain information about the class ('CG' or otherwise).
    repetitions (int): The number of repetitions for the evaluation.
    estimator (object): The machine learning estimator to use for training and prediction.

    Returns:
    None
    """
    
    subject_ids = list(set([key.split('-')[0] for key in dataset_dict.keys()]))  # Extract unique subject IDs
    # Scores
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

    for _ in tqdm(range(repetitions), desc='Repetitions'):
        loo = LeaveOneOut()
        for train_subject_idx, test_subject_idx in loo.split(subject_ids):  # Split subjects into training and test sets
            X_train, y_train = [], []
            X_test, y_test = [], []
            
            # Gather all samples for the training subjects
            train_subjects = [subject_ids[i] for i in train_subject_idx]
            for subject_id in train_subjects:
                for key, features in dataset_dict.items():
                    if key.startswith(subject_id):
                        X_train.append(features)
                        if 'CG' in key:
                            y_train.append(0)
                        else:
                            y_train.append(1) 
            
            # Gather all samples for the test subject
            test_subject_id = subject_ids[test_subject_idx[0]]
            for key, features in dataset_dict.items():
                if key.startswith(test_subject_id):
                    X_test.append(features)
                    if 'CG' in key:
                        y_test.append(0)
                    else:
                        y_test.append(1)
                    
            X_train, X_test = np.array(X_train), np.array(X_test)
            y_train, y_test = np.array(y_train), np.array(y_test)
            # print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
            
            # Train the model on the training set and make predictions on the test set
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            # Append metrics for this iteration
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    # Summarize metrics across all repetitions
    metrics_summary = pd.DataFrame({
        'Accuracy': [np.mean(accuracy_scores)], 
        'std Accuracy': [np.std(accuracy_scores)],
        'Precision': [np.mean(precision_scores)], 
        'std Precision': [np.std(precision_scores)],
        'Recall': [np.mean(recall_scores)], 
        'std Recall': [np.std(recall_scores)],
        'F1 Score': [np.mean(f1_scores)], 
        'std F1 Score': [np.std(f1_scores)],
    })
    
    display(metrics_summary)

def check_subject_video_count(dataset_dict):
    """
    Checks the number of videos associated with each subject in the dataset.
    Prints out any subjects that do not have exactly 8 videos and returns the number of errors found.

    Args:
        dataset_dict (dict): A dictionary where keys represent video identifiers in the format 'subjectID-videoID'.

    Returns:
        int: The number of subjects that do not have exactly 8 videos.
    """
    # Extract unique subject IDs
    subject_ids = list(set([key.split('-')[0] for key in dataset_dict.keys()]))
    
    error_count = 0
    
    # Loop through each subject and check the number of associated videos
    for subject in subject_ids:
        count_videos = len([key for key in dataset_dict.keys() if key.split('-')[0] == subject])
        
        if count_videos != 8:
            print(f"Subject {subject} has {count_videos} videos")
            error_count += 1
    if error_count == 0:
        print("All subjects have exactly 8 videos.")
    return error_count

def create_X_y_groups_arrays(dataset_dict):
    """
    Creates three arrays: X, y, and groups from the dataset dictionary.
    
    - X contains the values from the dataset dictionary.
    - y contains 1 if the key contains 'PG', and 0 if the key contains 'CG'.
    - groups contains the subject IDs extracted from the keys.
    
    Args:
        dataset_dict (dict): A dictionary where keys are strings in the format 'subjectID-videoID',
                             and values are the data points.

    Returns:
        tuple: A tuple of numpy arrays (X, y, groups)
            - X: Array of dataset values.
            - y: Array of labels, 1 for 'PG', 0 for 'CG'.
            - groups: Array of subject IDs.
    """
    X = []
    y = []
    ids = []
    groups = []
    
    # Iterate over each key, value pair in the dataset dictionary
    for key, value in dataset_dict.items():        
        X.append(value)        
        # Determine the label for y 
        if 'PG' in key:
            y.append(1)
        elif 'CG' in key:
            y.append(0)        
        # Extract the subject ID 
        ids.append(key)
        subject_id = key.split('-')[0]
        groups.append(subject_id)
        
    X = np.array(X)
    y = np.array(y)
    ids = np.array(ids)
    groups = np.array(groups)
    
    return X, y, ids, groups

def reduce_spd(covariance_to_reduce, n_components):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_to_reduce) # Calculate eigenvalues and eigenvectors            
    idx = eigenvalues.argsort()[::-1] # Sort eigenvalues in descending order
    eigenvectors = eigenvectors[:,idx][:,:n_components] # Sort eigenvectors according to eigenvalues and get the first n_components
    reduced_covariance = eigenvectors.T @ np.diag(eigenvalues) @ eigenvectors            
    return reduced_covariance
    
    
    
def reduce_spd_dataset(dataset_covs, n_components):
    dataset_covs = np.array(dataset_covs)
    new_dataset = np.zeros((dataset_covs.shape[0], dataset_covs.shape[1], n_components, n_components))
    for i in range(dataset_covs.shape[0]):
        for j in range(dataset_covs.shape[1]):
            covariance_to_reduce = dataset_covs[i, j]            
            reduced_covariance = reduce_spd(covariance_to_reduce, n_components)
            new_dataset[i, j] = reduced_covariance            
    return new_dataset    

