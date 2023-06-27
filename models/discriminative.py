# A discriminative model for supervised classification; Wrapped into the model interface given by BaseDisentangler.
# The model may be supported through partially annotated concepts.
from click import utils
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
from models.base.base_disentangler import BaseDisentangler
from architectures import classifiers
import common.constants as c
from tqdm import tqdm
from aicrowd.aicrowd_utils import evaluate_disentanglement_metric
from common.corr_utils import compute_binary_concept_labels_seven

class DiscriminativeModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)[0]

    def decode(self, z, **kwargs):
        raise NotImplementedError("Decoding is not supported for the discriminative model.")

    def forward(self, x, **kwargs):
        return self.encoder(x, **kwargs)


class Discriminative(BaseDisentangler):
    """
    A  Discriminative Classification model with labels and additionally annotated 
    ground truth factors if required.
    """

    def __init__(self, args):
        super().__init__(args)

        # hyper-parameters
        self.w_kld = args.w_kld
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)
        self.latent_reg = args.latent_reg
        print(args.labelled_idx)
        if args.labelled_idx is not None and args.labelled_idx[0] != -1: # Not none or -1
            self.indices_tensor = torch.tensor(args.labelled_idx).reshape(-1)
            self.n_annotated_factors = len(self.indices_tensor)
            # Function that computes the concepts from the ground truth factors.
            self.concept_fn = lambda x: x # compute_binary_concept_labels_seven if use 3dshapes (plain.)
        else:
            self.indices_tensor = None
            self.n_annotated_factors = None
            self.concept_fn = None

        # encoder and decoder
        classifier_name = args.classifier[0]
        my_classifier = getattr(classifiers, classifier_name)

        self.recon_iter = int(1e9) # never
        self.traverse_iter = int(1e9)
        self.float_iter =int(1e9)

        # model and optimizer
        self.model = DiscriminativeModel(my_classifier(self.z_dim, args.num_classes, n_annotated_factors=self.n_annotated_factors)).to(self.device)

        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }


    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu = self.model.encode(x=images)
        return mu

    def encode_stochastic(self, **kwargs):
        raise NotImplementedError("Stochastic encoding is not supported for the discriminative model.")

    def loss_fn(self, input_losses, **kwargs):
        z_latent = kwargs['z_latent']
        y_pred = kwargs['y_pred']
        y_true = kwargs['y_true']
        # Prediction for the annotated concepts
        c_pred = kwargs['c_pred'] if 'c_pred' in kwargs.keys() else None
        c_true = kwargs['c_true'] if 'c_true' in kwargs.keys() else None
        l1 = torch.nn.CrossEntropyLoss(reduction="sum")(y_pred, y_true.flatten()) 

        if c_true is not None:
            l2 = torch.nn.NLLLoss(reduction="sum")(torch.log_softmax(c_pred, dim=1), c_true.flatten())
        else:
            l2 = 0.0
        l3_reg = self.latent_reg*torch.mean(z_latent.pow(2))
        loss = l1 + self.w_kld*l2 + self.latent_reg*l3_reg
        return loss, l1, l2, l3_reg

    def discriminative_forward(self, losses, x_true1, label1, concept1= None):
        z, y_pred, c_pred = self.model.forward(x=x_true1)
        loss_fn_args = dict(z_latent=z, y_pred=y_pred, c_pred=c_pred, y_true=label1, c_true=concept1)
        losses[c.TOTAL_VAE], losses[c.CLASSIFICATION_LOSS], losses[c.CONCEPT_LOSS], losses[c.REG_LOSS]= self.loss_fn(losses, **loss_fn_args)
        return losses, loss_fn_args

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            vae_loss_sum = 0
            for internal_iter, (factors1, x_true1, label1) in enumerate(self.data_loader):
                losses = dict()
                if self.concept_fn is not None:
                    concepts = self.concept_fn(factors1)[:, self.indices_tensor].to(self.device)
                else:
                    concepts = None
                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device)
                losses, fn_args = self.discriminative_forward(losses, x_true1, label1, concept1=concepts)
                #print(losses)
                self.optim_G.zero_grad()
                losses[c.TOTAL_VAE].backward(retain_graph=False)
                self.optim_G.step()

                vae_loss_sum += losses[c.TOTAL_VAE].detach()
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum / internal_iter
                
                if not self.log_save(input_image=x_true1, recon_image=None, loss=losses):
                    print("ending_training.")
                    break
            # end of epoch
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        # Perform evaluation with disentanglement metrics here.
        logging.info('Using metrics:' +str(self.evaluation_metric))
        if self.evaluation_metric:
            self.evaluate_results = evaluate_disentanglement_metric(self, metric_names=self.evaluation_metric)
            print(self.evaluate_results)
            import json
            json.dump(self.evaluate_results, open(os.path.join(self.test_output_dir, "eval_results.json"), "w"))
        
        
