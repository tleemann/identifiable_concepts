# Utilities for inducing correlation in the ground truth data sets.
# We introduce correlations by changing the probabilites of the samples being drawn in the data loader.
import torch
import cv2
import numpy as np
import math

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    from: https://stackoverflow.com/a/43244194"""
    try:
        _ = torch.cholesky(B)
        return True
    except:
        return False


def get_normal_distr(sigma: float, n_cor: int, num_factor_levels: list, pairs: list):
    """
    Returns the density of a multivariate normal distribution where the first n_cor
    feature-tuples in pairs are correlated by sigma
    CAUTION: density is not normalized
    """
    n_cor = min(len(pairs), n_cor)

    mu = (torch.tensor(num_factor_levels, dtype=torch.float32) - 1) / 2
    cov = torch.diag(((mu + 0.5) / 2)**2) # We want the normal distribution variance to roughly cover all factor levels
    for feature1, feature2 in pairs[:n_cor]:
        cov[feature1, feature2] = sigma * torch.sqrt(cov[feature1, feature1] * cov[feature2, feature2])
        cov[feature2, feature1] = cov[feature1, feature2]

    # Ensure the covariance matrix is positive definite
    while not isPD(cov):
        # Reduce sigma until it is PD
        sigma = 0.9 * sigma
        for feature1, feature2 in pairs[:n_cor]:
            cov[feature1, feature2] = sigma * torch.sqrt(cov[feature1, feature1] * cov[feature2, feature2])
            cov[feature2, feature1] = cov[feature1, feature2]
    print("Using sigma={} to ensure positive definiteness of covariance matrix".format(sigma))

    cov_inv = cov.inverse()

    return (lambda x: torch.exp(-0.5 * torch.sum((x.float() - mu).matmul(cov_inv) * (x.float() - mu), dim=1)))


def correlation_map_line(line_width, corr_factor_sizes):
    """ Implementation of the correlation function taken over from Träuble et al. 
        Original code at:
        https://github.com/ftraeuble/disentanglement_lib/blob/39584ad5270a090723c8f12567c47d84f02444c9/
        disentanglement_lib/data/ground_truth/util.py
    """

    # Create a black image
    unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.uint8)

    width = math.ceil(line_width * min(corr_factor_sizes))

    offset = 0
    start = (0, offset)
    end = (corr_factor_sizes[1], corr_factor_sizes[0])

    kernel_width = min(corr_factor_sizes) // 4

    if not kernel_width % 2:  # kernels widths must be odd
        kernel_width += 1

    kernel_width_x = kernel_width
    kernel_width_y = kernel_width

    cv2.line(unnormalized_joint_prob, start, end, 255, width)

    unnormalized_joint_prob = cv2.GaussianBlur(unnormalized_joint_prob,
                                            (kernel_width_x, kernel_width_y), 0)
    unnormalized_joint_prob = torch.tensor(unnormalized_joint_prob, dtype=torch.float32)
    #print(unnormalized_joint_prob)
    return unnormalized_joint_prob

def sigma_correlation(sigma: float, feature1: int, feature2: int, dataset_str="shapes3d"):
    """ Return a lambda function which takes the ground truth factors as input, and returns the non-normalized density that
        that correlates two features (feature1, feature2). 
        sigma: Correlation strength 0.0 (Extreme correlation) -> inf (No correlation)
    """
    
    # Dataset sizes.
    num_factors = {"shapes3d": [10, 10, 10, 8, 4, 15],
    "shapes3d_binary": [8,8,8,8],
    "shapes3d_gaussian": [8,8,8,8],
    "shapes3d_noisy": [8,8,8,8],
    "dsprites_full": [3, 6, 40, 32, 32],
    "colorbar": [11, 11, 11]}

    if dataset_str in num_factors.keys():
        corr_map = correlation_map_line(sigma, (num_factors[dataset_str][feature1], num_factors[dataset_str][feature2]))
        if dataset_str != "shapes3d" and dataset_str != "dsprites_full" and dataset_str != "colorbar": # Some smoothing is required in this case (does not work for too small factor sizes)
            corr_map = torch.nn.AvgPool2d((4, 4), stride=(4, 4)).forward(corr_map.reshape(1,1,8,8))
            corr_map = corr_map.reshape(2,2)
            #print(corr_map)
        return lambda x: corr_map[x[:,feature1], x[:, feature2]]
    else:
        raise ValueError("Unsupported dataset. Please provide factor dimensions. ")

def filter_function(x):
    """ Filter out all samples with shape >=2 and no clear angle. """
    return (torch.abs(x[:, 5]-7) > 2).float()*(x[:, 4]<=1).float()

def compute_binary_concept_labels_seven(input_factors):
    """
        Compute the 4 binary concept labels that belong to to labels for the 
        function shapes3d_toy_labelfunction_seven.
        Dimension 0: Shape concept
        Dimension 1: Floor color
        Dimension 2: Object color
        Dimension 3: Orientation (left vs right)
    """
    return torch.stack(((input_factors[:,4] % 2 == 0).long(),
        (input_factors[:,0] < 5).long(),
        (input_factors[:,2] < 5).long(),
        (input_factors[:,5] <=7).long()), dim=1)
