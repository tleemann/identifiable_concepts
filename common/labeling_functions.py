import torch
from common.corr_utils import compute_binary_concept_labels_seven
# --- Labeling functions.

# Dict that assigns the number of classes for each label fn. We implement 2 labeling functions for 3dshapes, one with 7 labels and one with 8 labels.

labels_num_classes_dict = {
    "shapes3d_toy_labelfunction_seven": 7,
    "shapes3d_binary_labelfunction_eight": 8,
    "shapes3d_toy_labelfunction_eight": 8,
}

def shapes3d_toy_labelfunction_seven(input_factors, binary_factors = False):
    """ A label function for the 3dShapes data set with 7 classes (labeled 0-6).
        Classes: 0 = shape=1, Orentation < 0, object_hue =0-4
        Classes: 1 = shape=1, Orentation < 0, object_hue =0-4
        Classes: 2 = shape=1, Orentation > 0, object_hue =5-9
        Classes: 3 = shape=1, Orentation > 0, object_hue =5-9
        Classes: 4 = shape=0
        Filter out: shape = 0,4, orientation = 0
        Concepts: Wall hue (4 vs all), orientation left/right (1,3 vs 0,2), shape (1,2 vs 3,4)
    """
    if len(input_factors.shape) == 1:
        input_factors = input_factors.reshape(1,-1) 

    label = (input_factors[:,5]>7).long()*2 + (input_factors[:,2] < 5).long() # classes 0-3
    label[input_factors[:,4] % 2 == 0]=4+2*(input_factors[input_factors[:,4] % 2 == 0, 0] < 5).long() # classes 4,6
    label[label==6] = label[label==6]-(input_factors[label==6,5]<=7).long()
    return label.reshape(-1)

def shapes3d_binary_labelfunction_eight(input_factors):
    """ Use this function to assign the class labels 0-7 on 3dshapes_binary. """
    return shapes3d_toy_labelfunction_eight(input_factors, True)

def shapes3d_toy_labelfunction_eight(input_factors, binary_factors = False):
    """ A label function for the 3dShapes data set with 8 classes (labeled 0-7).
        See notebooks/DecisionTree8.png for a visualization of the distinct labels that are assingned by this function.
    """
    if len(input_factors.shape) == 1:
        input_factors = input_factors.reshape(1,-1) 
    if not binary_factors:
        concepts = compute_binary_concept_labels_seven(input_factors)
    else:
        concepts = input_factors
    
    label = torch.zeros(len(concepts), dtype=torch.long)
    # Left branch
    bl = concepts[:, 0] == 0 # left branch
    br = concepts[:, 0] == 1 # right branch
    label[bl] = 2*concepts[bl, 3] + concepts[bl, 2]
    label[br] = 4 + 2*concepts[br, 1] + concepts[br, 3]
    return label.reshape(-1)
