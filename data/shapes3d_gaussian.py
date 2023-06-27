# A 3dshapes dataset with only 10% of its size for local debugging, acting like the real shapes3d dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import os
import torch

class Shapes3DGaussian(ground_truth_data.GroundTruthData):
    """ Shapes3D binary dataset with an underlying gaussian distribution.
        This is a variant of the 3d-Shapes data set with binary attributes. The dataset files need
        to be generated previously and stored on disk to be able to use this class.
    The data set was originally introduced in "Disentangling by Factorising".

    The ground-truth factors of variation are:
    0 - object shape (2 different values, 0 = Cylinder, 1 Cube)
    1 - wall color (2 different values, 0 = Blue/Red, 1 = Yellow/Green)
    2 - object color (2 different values, 0 = Blue/Red, 1 = Yellow/Green)
    3 - orientation (azimzut) (2 different values 0 = left, 1 = right)

    Note that in this dataset, there can be multiple images with the same ground
    truth FoV.
    """

    def __init__(self, name="gaussian"):
        """ 
            pass name=gaussian: for resampled dataset with perfect factor labels
            pass name=noisy for resampled dataset with noisy factor labels
            Both datasets have 4 binary factors of variation (components)
        """
        self.images = np.load(os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), f"3dshapes_{name}", f"3dshapes_{name}_images.npy"))
        labels = np.load(os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), f"3dshapes_{name}", f"3dshapes_{name}_labels.npy"))

        n_samples = self.images.shape[0]
        if name == "gaussian":
            features = labels.reshape([n_samples, 6])
        else:
            features = labels.reshape([n_samples, 4]) # 3dshapes noisy only has 4 factors when loaded from disk.
        self.factor_sizes = [2, 2, 2, 2]


        self.latent_factor_indices = list(range(4))
        self.num_total_factors = features.shape[1]
        #self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        if name == "gaussian":
            factors = np.stack(((labels[:,4] % 2 == 0),
                (labels[:,0] < 0.5),
                (labels[:,2] < 0.5),
                (labels[:,5] < 0.0)), axis=1)
        else:
            factors = labels.astype(int)

        factor_num = 8*factors[:,0] + 4*factors[:,1] + 2*factors[:,2] + factors[:,3]
        unique, self.factor_counts = np.unique(factor_num, return_counts=True)
        assert len(unique) == 16
        #print("Factor counts", self.factor_counts)
        self.factors_offset = np.concatenate((np.array([0]), np.cumsum(self.factor_counts)))
        #print("Factor Offsets", self.factors_offset)
        
    @property
    def num_factors(self):
        return len(self.factor_sizes)

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [64, 64, 3]

    def sample_factors(self, num, random_state=0):
        """ Sample a batch of factors Y.
            All factors have an equal probability of 0.5 of occurance.
        """
        return (np.random.rand(num, len(self.factor_sizes)) > 0.5).astype(np.int)

    def sample_observations_from_factors(self, factors, random_state=0):
        factors_idx = 8*factors[:,0] + 4*factors[:,1] + 2 *factors[:, 2] + factors[:, 3]
        factors_offsets = self.factors_offset[factors_idx]
        factors_offsets += (np.random.rand(len(factors))*self.factor_counts[factors_idx]).astype(np.int)
        return self.images[factors_offsets]
    
    def __len__(self):
        return len(self.images)
