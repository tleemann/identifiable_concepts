import torch.nn as nn
import torch.nn.functional as func
from common.utils import init_layers
from common.ops import Flatten3D

class LatentExtractor(nn.Module):
    """ A simple encoder that maps an image to latent dimensions z. On this latent representation,
        further downstream tasks can be conducted, e.g. classification.
    """
    def __init__(self, dim_latent: int,input_size=64):
        """ Init the network object.
         Parameters:
             dim_latent: Number of latent space dimensions z

        """
        self.dim_latent = dim_latent
        super(LatentExtractor, self).__init__()

        # Define the layers here
        # The encoder
        # Parameters of the decoder
        # For mean and variance
        self.layer1_channels = 64
        self.layer2_channels = 128
        #self.num_interm_features = 16
        self.conv1 = nn.Conv2d(3, self.layer1_channels, 3)
        #self.conv1b = nn.Conv2d(1, self.layer1_channels, 3)
        shape = input_size - 2
        self.pool = nn.MaxPool2d((2, 2))
        shape = shape // 2
        self.conv2 = nn.Conv2d(self.layer1_channels, self.layer2_channels, 3)
        shape = shape - 2
        shape = shape // 2
        #print(shape)
        self.last_latent_shape = shape
        #self.denseInterm = nn.Linear(self.layer2_channels*self.last_latent_shape*self.last_latent_shape, self.num_interm_features)
        self.final = nn.Linear(self.layer2_channels, self.dim_latent) # Output for mean

    # Forward method
    def forward(self, x):
        # Encode
        z_mean = self.encode(x)
        return z_mean 

    def encode_conv(self, x_in):
        """ Run only convolutional part until Global Average Pooling. """
        x = func.relu(self.conv1(x_in))
        #y = func.relu(self.conv1b(x_in.mean(dim=1, keepdim=True)))
        x = self.pool(x)
        #x = self.pool(torch.cat((x,y),dim=1))
        x = func.relu(self.conv2(x))
        return x
    
    # Return tensor of encoded vectors
    # x: Image, y class as one hot encoded tensor
    def encode(self, x_in):
        x = self.encode_conv(x_in)
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2) # Change to global average pooling.
        #x = self.pool(x)
        #x = self.denseInterm(x.reshape(len(x), -1))
        return self.final(x) #.reshape(len(x), -1)))


class CategoricalLabelPredictor(nn.Module):
    """ Simple dense layer for prediction of categorical class label from 
        a continuous input. Returns logits.
    """
    def __init__(self, input_dim=3, n_cats = 5):
        """ Init the network object.
        Parameters:
            input_dim: Number of input dimensions z, e.g. concepts or a latent vectors
            n_cats: Number of output logits (corresponding to the categories)
        """
        self.input_dim = input_dim
        self.n_cats = n_cats
        super(CategoricalLabelPredictor, self).__init__()

        self.dense2 = nn.Linear(self.input_dim, self.n_cats )

    # Forward method concepts-> labels.
    def forward(self, d):
        # Encode
        z_mean = self.dense2(d)
        return z_mean

class ConceptObservationPredictor(nn.Module):
    """ Simple MLP to predict concepts from a latent representation. """
    def __init__(self, input_dim=3, output_dim=2):
        """ Init the network object.
         Parameters:
             input_dim: Number of latent space dimensions z
             output_dim: Number of concepts
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(ConceptObservationPredictor, self).__init__()
        #self.dense1 = nn.Linear(self.input_dim, self.dim_latent)
        self.dense2Mu = nn.Linear(self.input_dim, 2*output_dim)
        #self.dense2Sig = nn.Linear(self.dim_latent, output_dim)
        
    # Forward method concepts-> observations.
    def forward(self, d):
        # Encode
        #interm_rep = torch.celu(self.dense1(d))
        z_mean = self.dense2Mu(d).reshape(-1,self.output_dim,2)
        #z_log_sigma = self.dense2Sig(interm_rep)
        return z_mean #, z_log_sigma

class SimpleClassifier(nn.Module):
    def __init__(self, z_dim, n_cats=5, n_annotated_factors=None, l1_fact = 1e-4, grad_pen = 1e-4):
        """ Init the network object.
         Parameters:
             n_annotated_concepts
             n_total_concepts
        """
        super(SimpleClassifier, self).__init__()
        print(f"Initializing Simple Classifier with zdim={z_dim} and {n_cats} classes.")
        self.z_extr = LatentExtractor(dim_latent = z_dim) # x->z (latent representation)
        self.y_pred = CategoricalLabelPredictor(input_dim = z_dim, n_cats = n_cats) # d->y
        if n_annotated_factors is not None:
            self.c_pred = ConceptObservationPredictor(input_dim = z_dim, output_dim=n_annotated_concepts) # d->c
            self.n_annotated_concepts = n_annotated_concepts
        else:
            self.c_pred = None
        self.z_dim = z_dim
        self.n_cats = n_cats
        self.l1fact = l1_fact
        self.grad_pen = grad_pen

    def forward(self, x, **kwargs):
        """ Forward the model with a batch of data.
            Return the (latent embeddings, the label logits) for a model without annotated concepts,
            Return the (latent embeddings, the label logits, and the concept logits) for a model with annotated concepts.
        """
        z = self.z_extr(x)
        yp = self.y_pred(z)
        if self.c_pred is None:
            return z, yp, None
        else:
            return z, yp, self.c_pred(z).reshape(-1,2)

class SimpleConv64Extractor(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, latent_dim, 4, 2, 1),
            Flatten3D(),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)
    
class GAPClassifier(nn.Module):
    def __init__(self, z_dim, n_cats=5, n_annotated_factors=None, l1_fact = 1e-4, grad_pen = 1e-4):
        """ A discriminative classifier with an intermediate concept representation using a convoluational extractor.
         Parameters:
             z_dim: Number of latent dimensions of the concept representation.
             n_annotated_factors: Number of ground truth factors that are annotated. pass None, if no concepts are available.
                    If a number is passed, a ground truth predictor for the concepts will be initialized.
             n_cats: Number of classes in the classification problem
        """
        super().__init__()
        print(f"Initializing GAP Classifier with zdim={z_dim}, {n_cats} classes and {n_annotated_factors} annotated concepts.")
        self.z_extr = SimpleConv64Extractor(latent_dim = z_dim, num_channels=3, image_size=64) # x->z (latent representation)
        self.y_pred = CategoricalLabelPredictor(input_dim = z_dim, n_cats = n_cats) # z->y (predicts the label)
        if n_annotated_factors is not None:
            self.c_pred = ConceptObservationPredictor(input_dim = z_dim, output_dim=n_annotated_factors) # z->c (predicts the ground truth concepts)
            self.n_annotated_concepts = n_annotated_factors
        else:
            self.c_pred = None
        self.z_dim = z_dim
        self.n_cats = n_cats
        self.l1fact = l1_fact
        self.grad_pen = grad_pen

    def forward(self, x, **kwargs):
        """ Forward the model with a batch of data.
            Return the (latent embeddings, the label logits) for a model without annotated concepts,
            Return the (latent embeddings, the label logits, and the concept logits) for a model with annotated concepts.
        """
        z = self.z_extr(x)
        yp = self.y_pred(z)
        if self.c_pred is None:
            return z, yp, None
        else:
            return z, yp, self.c_pred(z).reshape(-1,2)