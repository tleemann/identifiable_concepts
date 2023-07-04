# Implementation of the toy datasets.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import os
import torch
import math

class ColorBar(ground_truth_data.GroundTruthData):
    """ A toy dataset with a bar that can change its width, position and color.
        This dataset is referred to as ColorBar in the paper in shown in Figure 3b.
        FoV:
            (1) Color
            (2) Position (up / down)
            (3) Width (broad / narrow)
    """

    def __init__(self, numsteps, nonlin_colors=True, cont_gaussian=False):
        """ Initialize a new toy bar dataset. 
            numsteps provides the number of factor values. Passing N will result in a valid range from
            [0, N-1], which can be sampled continuously.
            nonlin_colors: boolean to indicate if the color change should happen at a uniform speed or not.
                if it happens at uniform space (nonlin_colors=False), some factors may not be identifiable.
        """
        A = ([numsteps]*3)
        self.factors_size = A
        self.numsteps = numsteps
        self.images = np.zeros((16384, 3, 8, 8)) # Dummy images.
        self.nonlin_colors = nonlin_colors
        self.cont_gaussian = cont_gaussian

    @property
    def num_factors(self):
        return len(self.factors_size)

    @property
    def factors_num_values(self):
        return self.factors_size

    @property
    def observation_shape(self):
        return [8, 8, 3]

    def sample_factors(self, num, random_state=0):
        """ Sample a batch of factors Y.
            All factors have an equal probability of 0.5 of occurance.
        """
        if self.cont_gaussian:
            simple_fact =  (0.5+0.2*(np.random.randn(num, len(self.factors_size))))*(self.numsteps-1)
            simple_fact[simple_fact<0.0] = 0.0
            simple_fact[simple_fact>(self.numsteps-1)] = self.numsteps-1
            return simple_fact
        else:
            return ((np.random.rand(num, len(self.factors_size)))*self.numsteps).astype(np.int)

    def sample_observations_from_factors(self, factors, random_state=0, ret_torch=False):
        if type(factors) == np.ndarray:
            factors = torch.from_numpy(factors)

        base_colors = factors.float()/(self.numsteps-1+1e-2) # normalize the factors.
        
        
        colors = torch.zeros(len(factors), 3, device=factors.device)
        if self.nonlin_colors:
            fbc = (torch.exp(base_colors[:, 0])-1.0)/(math.exp(1.0)-1.0) # non-lin. mapping from 0-1 to 0-1 with non-zero gradient 
            colors[:, 0] += torch.sin(fbc*math.pi*0.5)
            colors[:, 2] += torch.cos(fbc*math.pi*0.5)
        else:
            colors[:, 0] += torch.sin(base_colors[:, 0]*math.pi*0.5)
            colors[:, 2] += torch.cos(base_colors[:, 0]*math.pi*0.5)
        colors[:, 1] = 1.0
        width = base_colors[:, 1]*3.0
        height = base_colors[:, 2]*6.0


        int_height = height.long()
        int_width = width.long()
        forward_height = torch.sin((height-int_height.float())*math.pi*0.5)
        backward_height = torch.cos((height-int_height.float())*math.pi*0.5)
        forward_width = width-int_width.float()
        #green = torch.tensor([0.0, 1.0, 0.0])
        #print("H", int_height, forward_height)
        #print("W", int_width, forward_width)
        sample_matrix = torch.ones([len(factors)]+self.observation_shape, device=factors.device) #*green.reshape(1,1,1,3)
        for i in range(len(factors)):
            # Main bar.
            sample_matrix[i, int_height[i]+1, 3-int_width[i]:5+int_width[i],  :] -= colors[i, :]

            # Width borders
            #m = torch.tensor([1.0, 0.0, 0.0]).reshape(-1,1)
            ww = torch.cat((backward_height[i].reshape(-1), torch.ones(1, device=factors.device), forward_height[i].reshape(-1))).reshape(-1,1) 
            
            #print(ww)
            #print(torch.cat((backward_height[i].reshape(-1), torch.ones(1), forward_height[i].reshape(-1))).reshape(-1,1)*colors[i, :].reshape(1,-1))
            
            sample_matrix[i, (int_height[i]):(int_height[i]+3), 3-int_width[i]-1, :] -= forward_width[i]*ww*colors[i, :]
            sample_matrix[i, (int_height[i]):(int_height[i]+3), 5+int_width[i], :] -= forward_width[i]*ww*colors[i, :]

            #sample_matrix[i, int_height[i]+1, 3-int_width[i]-1, :] -= forward_width[i]*colors[i, :]
            #sample_matrix[i, int_height[i]+1, 5+int_width[i], :] -= forward_width[i]*colors[i, :]

            #sample_matrix[i, int_height[i]+1, 3-int_width[i]-1, :] -= forward_width[i]*colors[i, :]
            #sample_matrix[i, int_height[i]+1, 5+int_width[i], :] -= forward_width[i]*colors[i, :]

            sample_matrix[i, int_height[i],  3-int_width[i]:5+int_width[i], :] -= backward_height[i]*colors[i, :]
            sample_matrix[i, int_height[i]+2,  3-int_width[i]:5+int_width[i], :] -= forward_height[i]*colors[i, :]

            # Height moving bars.
            #sample_matrix[i, int_height[i], 3-int_width[i]:5+int_width[i], :] -= (1.0-forward_height[i])*(1.0-colors[i, :])
            #sample_matrix[i, int_height[i]+2, 3-int_width[i]:5+int_width[i], :] -= forward_height[i]*(1.0-colors[i, :])

        return sample_matrix if ret_torch else sample_matrix.numpy()


class FourBars(ground_truth_data.GroundTruthData):
    """ FourBars toy dataset (shown in Figure 3a).
        Each FoV is a bar that moves up and down while changing colors from white to black.
    """

    def __init__(self, numsteps, non_linear=False):
        A = ([numsteps]*4)
        self.factors_size = A
        self.numsteps = numsteps
        self.images = np.zeros((16384, 3, 8, 8)) # Dummy images.
        self.non_linear = non_linear
    @property
    def num_factors(self):
        return len(self.factors_size)

    @property
    def factors_num_values(self):
        return self.factors_size

    @property
    def observation_shape(self):
        return [8, 8, 3]

    def sample_factors(self, num, random_state=0):
        """ Sample a batch of factors Y.
            All factors have an equal probability of 0.5 of occurance.
        """
        return ((np.random.rand(num, len(self.factors_size)))*self.numsteps).astype(np.int)

    def sample_observations_from_factors(self, factors, random_state=0, ret_torch=False):
        if type(factors) == np.ndarray:
            factors = torch.from_numpy(factors)

        base_colors = factors.float()/(self.numsteps-1)
        if self.non_linear:
            # Non-linear monotonous [0 - 1] to [0 - 1] mapping
            bases = 2.0*math.pi*torch.tensor([0,0.5,1.0,1.5]).reshape(1, -1)
            base_colors = base_colors + 0.1*torch.sin(factors*bases)

        #print(base_colors[:,3])
        sample_matrix = torch.ones([len(factors)]+self.observation_shape)*0.5
        
        
        #print(sample_matrix.shape)
        center_loc = torch.tensor([0, 3, 6, 0], dtype=torch.long)
        for i in range(len(factors)):
            #print(center_loc, right_weight)
            linestarts = [0, 0, 0, 5] # left/right offset.
            lineends = [4, 4, 4, 7] # left/right offset.
            for sq in range(4):
                if sq == 3:
                    bcnorm = 4.0*(base_colors[i,sq]-0.5)/np.sqrt(2.0) + 3.0
                    #print(bcnorm)
                    center_loc_offset = bcnorm.long()
                    right_weight = bcnorm - center_loc_offset
                    left_weight = 1.0 - right_weight
                    base_colors_line = 1.0
                    #print(center_loc_offset, right_weight, left_weight)
                    sample_matrix[i, center_loc[sq]+center_loc_offset-1, linestarts[sq]:lineends[sq]] += 0.5*left_weight
                    sample_matrix[i, center_loc[sq]+center_loc_offset+2, linestarts[sq]:lineends[sq]] += 0.5*right_weight
                else:
                    center_loc_offset = 0
                    base_colors_line = base_colors[i, sq]
                
                sample_matrix[i, center_loc[sq]+center_loc_offset:center_loc[sq]+center_loc_offset+2, linestarts[sq]:lineends[sq]] += (base_colors_line - 0.5)
                #print(sample_factors[i])
        return sample_matrix if ret_torch else sample_matrix.numpy()

    def __len__(self):
        return len(self.images)
