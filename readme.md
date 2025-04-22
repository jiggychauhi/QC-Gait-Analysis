This repository contains the source code and datasets associated with the research paper entitled:

**"Parkinson Disease Classification: A Comparison of Quantum and RBF Kernels Using Support Vector Machine"**

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

    @inproceedings{olmos2022gait,
    title={Gait patterns coded as Riemannian mean covariances to support Parkinson’s disease diagnosis},
    author={Olmos, Juan and Galvis, Juan and Mart{\'\i}nez, Fabio},
    booktitle={Ibero-American Conference on Artificial Intelligence},
    pages={3--14},
    year={2022}
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
