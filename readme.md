# Source code of the paper "Gait patterns coded as Riemannian mean covariances to support Parkinson’s disease diagnosis"

In the folder ``data`` are the two datasets used in the paper. For example ``covs_DN20`` corresponds to frame covariance matrices calculated from DenseNet features. The nomenclature CG_XX_YY.npy correspond in this case for a control subject, patient XX, and video YY. 

In the notebook ``evaluation_covariance_mean_svm`` is presented the result of copmute the covariance mean of videos as described in the paper, that is, computing the Algorithm1 (Presented in the paper) to calculate the covariance mean of frame covariances. Then, the covariance means of each video is projected to the tangent space on the Identity matrix, and the upper triangular part of those matrices is taken as feature vector for each sample. From these vectors is computed the SVM. 



This repository contains the source code and datasets associated with the research paper entitled:

**"Gait Patterns Coded as Riemannian Mean Covariances to Support Parkinson’s Disease Diagnosis"**

## Authors
- Juan Olmos<sup>1</sup>, Juan Galvis <sup>2</sup>, Fabio Martínez <sup>1*</sup>


## Datasets
The `data` folder contains two datasets utilized in the research:
- `covs_DN20`: This dataset consists of frame covariance matrices calculated using DenseNet features.
- The files named `CG_XX_YY.npy` represent data for a control subject, identified by patient XX and video YY.

## Notebook
The Jupyter notebook `evaluation_covariance_mean_svm` presents the methodology described in the paper. This includes:
1. Computing the Riemannian mean of the covariance matrices of video frames as outlined by Algorithm 1 in the paper.
2. Projecting the covariance means onto the tangent space at the identity matrix.
3. Extracting the upper triangular part of these matrices to serve as feature vectors for SVM classification.

## Citation
Please cite our work if it helps your research:

    @article{Olmos2023Gait,
      title={Gait Patterns Coded as Riemannian Mean Covariances to Support Parkinson’s Disease Diagnosis},
      author={Olmos, Juan and Manzanera, Antoine and Martínez, Fabio},
      journal={Journal Title},
      year={2023},
    }

## Project Structure
    ├── data
    │   └── ...                        <- Dataset folders and files.
    │
    ├── evaluation_covariance_mean_svm.ipynb
    |
    ├── README.md                      <- The top-level README for developers using this project.
    |
    └── src                            <- Source code for the implementation.

## Contact Information
- **Juan A. Olmos**: jaolmosr@correo.uis.edu.co

For further information, contributions, or questions, feel free to contact the authors.
