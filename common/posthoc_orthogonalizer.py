# post-hoc orthogonalization of latent spaces to discover meaningful components.
from cProfile import run
from unittest import runner
from attr import attributes
from torch.autograd import grad
import torch 
from tqdm import tqdm
import math
from torch.optim import Adam, RMSprop

from common.utils import prepare_data_for_visualization
from common.analysis import gt_get_binary_score_matrix
from torchvision.utils import save_image, make_grid

import os, sys
import common.constants as c

import matplotlib.pyplot as plt
import numpy as np


class LinearTransformedNet_Stub():
    # This class wraps a linear transform matrix around a VAE without changing the VAE inplace
    # It is solely to be used as input for the aicrowd_utils.evaluate_disentanglement_metric function, 
    # so that we can use the common interface

    class ModelStub():
        def __init__(self):
            self.encoder = 0
            self.decoder = 0

    class EncoderStub(torch.nn.Module):
        def __init__(self, base_encoder, transform):
            super().__init__()

            self.transform = transform.cpu()
            self.base_encoder = base_encoder

        def forward(self, x):
            res = self.base_encoder(x)
            mu, logvar = res[0], res[1]
            mu = mu.cpu()
            mu = self.transform.matmul(mu.unsqueeze(2)).squeeze(2)
            return mu, logvar


    def __init__(self, base_model, transform):
        self.base_model = base_model
        self.transform = transform.to(self.base_model.device)
        if self.transform.size(0) == self.transform.size(1):
            self.inv_transform = torch.inverse(self.transform)
        else:
            self.inv_transform = None

        # Setup the encoder and decoder with attached matrices
        self.model = self.ModelStub()
        self.model.encoder = self.EncoderStub(self.base_model.model.encoder, self.transform)

        if self.inv_transform is not None:
            self.inv_transform_layer = torch.nn.Linear(self.inv_transform.size(0), self.inv_transform.size(1), bias=False)
            self.inv_transform_layer.weight = torch.nn.Parameter(self.inv_transform)
            self.model.decoder = torch.nn.Sequential(self.inv_transform_layer, self.base_model.model.decoder)

        # Other attributes that aicrowd_utils.evaluate_disentanglement_metric function uses
        self.num_channels = base_model.num_channels
        self.image_size = base_model.image_size
        self.ckpt_dir = base_model.ckpt_dir
        self.dset_name = base_model.dset_name


class LinearTransformedNet():
    def __init__(self, base_model, transform):
        self.base_model = base_model
        self.transform = transform.to(self.base_model.device)
        if self.transform.size(0) == self.transform.size(1):
            self.inv_transform = torch.inverse(self.transform)
        else:
            self.inv_transform = None
        print(self.transform.device)

    @property
    def device(self):
        return self.base_model.device
    
    def encode_deterministic(self, **kwargs):
        # print(self.base_model.encode_deterministic(**kwargs).shape)
        return self.transform.matmul(self.base_model.encode_deterministic(**kwargs).unsqueeze(2)).squeeze(2) #[B, 8, 1]

    def decode(self, **kwargs):  
        kwargs["latent"] = self.inv_transform.matmul(kwargs["latent"].unsqueeze(2)).squeeze(2)
        #print(kwargs["latent"].shape)
        return self.base_model.decode(**kwargs)

    @property
    def z_dim(self):
        return self.transform.size(0)

    @property
    def traverse_min(self):
        return self.base_model.traverse_min
    
    @property
    def traverse_max(self):
        return self.base_model.traverse_max
    
    @property
    def traverse_spacing(self):
        return self.base_model.traverse_spacing

    @property
    def train_output_dir(self):
        return self.base_model.train_output_dir
    
    @property
    def test_output_dir(self):
        return self.base_model.test_output_dir

    @property
    def ckpt_dir(self):
        return self.base_model.ckpt_dir

    @property
    def model(self):
        return self.base_model.model

    @property
    def num_channels(self):
        return self.base_model.num_channels
    
    @property
    def image_size(self):
        return self.base_model.image_size

    @property
    def test(self):
        return self.base_model.test

    def visualize_traverse(self, limit: tuple, spacing, data, test=False, z_dirmat = None):
        self.base_model.net_mode(train=False)
        interp_values = torch.arange(limit[0], limit[1], spacing, device = self.device)
        sample_images_dict, sample_labels_dict = prepare_data_for_visualization(data)

        encodings = dict()
        # encode original images.
        for key in sample_images_dict.keys():
            encodings[key] = self.encode_deterministic(images=sample_images_dict[key], labels=sample_labels_dict[key])

        for key in encodings:
            latent_orig = encodings[key]
            label_orig = sample_labels_dict[key]
            #logging.debug('latent_orig: {}, label_orig: {}'.format(latent_orig, label_orig))
            samples = []

            # encode original on the first row
            sample = self.decode(latent=latent_orig.detach(), labels=label_orig)
            #for _ in interp_values:
            #    samples.append(sample)

            num_cols = interp_values.size(0)
            num_dims = (z_dirmat.size(1) if z_dirmat is not None else self.z_dim)
            print(latent_orig.shape)
            for zid in range(num_dims):
                for val in interp_values:
                    latent = latent_orig.clone()
                    if z_dirmat is not None:
                        latent += val*z_dirmat[:, zid].to(self.device)
                    else:
                        latent[:, zid] =  latent_orig[:, zid] + val
                    sample = self.decode(latent=latent, labels=label_orig).detach()
                    samples.append(sample)

            print(len(samples))
            if test:
                file_name = os.path.join(self.test_output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, self.iter, key, c.JPG))
            else:
                file_name = os.path.join(self.train_output_dir, '{}_{}.{}'.format(c.TRAVERSE, key, c.JPG))
            samples = torch.cat(samples, dim=0).cpu()
            samples = make_grid(samples, nrow=num_cols)
            save_image(samples, file_name)


# def batch_optimize_orthogonality(net, testloader, A_init, batches=int(1e8), 
#         attribution_fn = input_jacobian, ground_truth_dirs=None, eval_interval = 20):
#     """ Optimize the matrix A_init such that the directions are orthogonalized. """
#     A = A_init.clone().to(net.device)
#     A.requires_grad_(True)
#     adam = Adam((A,), lr=1e-5)
#     running_loss = 0.0
#     stats = []
#     with tqdm(total=len(testloader), position=0, leave=True) as pbar:
#         for i, data in tqdm(enumerate(testloader)):
#             _, x, _ = tuple([x.to(net.device) for x in data])
#             J = attribution_fn(net, x).detach().contiguous()
#             #print(J.grad_fn)
#             J_orgshape = J.shape
#             J = J.view(J.size(0), J.size(1), -1) # [B, latent_dim, input_dim]
#             disentangledJ = torch.matmul(A.unsqueeze(0), J)
#             loss = torch.sum(disjoint_orthogonality_loss(disentangledJ.reshape(J_orgshape))[1])
#             loss.backward()
#             adam.step()
#             running_loss += loss.detach().item()
#             pbar.update(1)
#             if i >= batches:
#                 break   
#             if i % eval_interval == (eval_interval -1) and ground_truth_dirs is not None:
#                 dir_score =  change_matrix_disentanglement(A.detach().cpu(), ground_truth_dirs)
#                 print("Disentanglement score: ", dir_score)
#                 print("Running loss: ", running_loss/eval_interval)
#                 stats.append({"iter": i, "oloss": running_loss/eval_interval, "dirscore": dir_score})
#                 running_loss = 0.0
#         print(running_loss)
#     return A.detach(), stats


def disjoint_orthogonality_loss(attrs, normalize=True, disjoint=True, losstype="offdiagl2"):
    """ Loss on disjointness of attributions. 
        attrs Input: Shape [B, Z, C, H, W]
        normalize: Normalize attributions to one.
        perpixel: have on attribution for each pixel
        disjoint: Take the absolute values of the attribution first so that for orthogonality also disjointness is required.
    """
    if disjoint:
        nattrs = torch.abs(attrs)
    else:
        nattrs = attrs

    nattrs = nattrs.reshape(len(attrs), attrs.size(1), -1) # [B, Z, C*H*W]

    if normalize:
        nattrs = nattrs / nattrs.norm(dim=2, p=1, keepdim=True)

    orthogonality = torch.bmm(nattrs, nattrs.transpose(1,2)) # [B, Z, Z]

    if losstype=="offdiagl2": # The loss described in the paper.
        orthogonality = orthogonality - torch.diag_embed(torch.diagonal(orthogonality, offset=0, dim1=1, dim2=2))
        oloss = torch.sqrt(torch.sum(orthogonality.reshape(len(attrs), -1).pow(2), dim=1))

    elif losstype=="detloss":
        tensor_list = []
        for i in range(len(orthogonality)):
            prod = torch.prod(torch.abs(torch.diag(orthogonality[i])))
            tensor_list.append((prod-torch.det(orthogonality[i]))/prod)
        oloss = torch.stack(tensor_list)

    elif losstype=="logdetloss":
        tensor_list = []
        for i in range(len(orthogonality)):
            prod = torch.sum(torch.log(torch.abs(torch.diag(orthogonality[i]))))
            tensor_list.append(prod-torch.logdet(orthogonality[i]))
        oloss = torch.stack(tensor_list)
    return orthogonality, oloss

# def find_B_with_chol(orth_1, orth_2):
#     """ 
#         Compute the analytical soluntion for B, using orth_1 = JJ^T at point 1 and orth_2 = JJ^T at point 2.
#         This algorithm is described in Supplementary B4 in the paper.
#     """
#     #if True:
#     #    orth_1 = torch.sum(orth_1, dim=0).unsqueeze(0)
#     # Obtain U via Cholesky
#     # Rule out singular orth_matrices.
#     Gamma_b = orth_2
#     if len(orth_1.shape) > 2:
#         U_arr = []
#         non_sing_index = torch.ones(len(orth_1), dtype=torch.long)
#         for i in range(len(orth_1)):
#             if torch.isnan(orth_1[i]).any():
#                 non_sing_index[i] = 0
#                 continue
#             Dslash, V = torch.symeig(orth_1[i], eigenvectors=True)
#             if Dslash[0] < 1e-8:
#                 non_sing_index[i] = 0
#             else:
#                 U_arr.append(torch.diag(1 / torch.sqrt(Dslash)) @ V.t())
#         U = torch.stack(U_arr, dim=0)
#         non_sing_items = torch.sum(non_sing_index)
#         if torch.sum(non_sing_index) < len(orth_1):
#             print("Dopping" , len(orth_1) - non_sing_items, "singular orthogonality matrices.", file=sys.stderr)
#             Gamma_b = orth_2[non_sing_index==1]
#     else:
#         U = torch.inverse(torch.cholesky(orth_1))
#     # Obtain U via eignevalues:
#     #U = torch.zeros(orth_1.shape)
#     #for i in range(U.shape[0]):
#     #    E, D = torch.symeig(orth_1[i], eigenvectors=True)
#     #    U[i] = torch.diag(1 / torch.sqrt(E)) @ D.t()
#     print(U.shape, Gamma_b.shape)
#     Gamma = U @ Gamma_b @ U.transpose(-2, -1)
#     eigenvector_ratios = []
#     if len(Gamma.shape) > 2:
#         Q_arr = []
#         for i in range(len(Gamma)):
#             try:
#                 Dslash, Q = torch.symeig(Gamma[i], eigenvectors=True)
#             except RuntimeError:
#                 print(Gamma[i])
#                 continue
#             Q_arr.append(Q)
#             eigenvector_ratios.append(Dslash[-1]/Dslash[0])
#         Q = torch.stack(Q_arr, dim=0)
#     else:
#         Dslash, Q = torch.symeig(Gamma, eigenvectors=True)
#         eigenvector_ratios.append(Dslash[-1]/Dslash[0])
#     #print(Q.t() @ Gamma @ Q)
#     #print("Found gradient norm ratios:", Dslash)
#     return Q.transpose(-2, -1) @ U, eigenvector_ratios

# def compute_orthogonality_matrices(nattrs, normalize=False):
#     nattrs = nattrs.reshape(len(nattrs), nattrs.size(1), -1) # [B, Z, C*H*W]
#     #print(nattrs.shape)
#     if normalize:
#         nattrs = nattrs / nattrs.norm(dim=2, keepdim=True)
#     orthogonality = torch.bmm(nattrs, nattrs.transpose(1,2)) # [B, Z, Z]
#     return orthogonality
    
# def iterative_eigendecomposition(init_matrix, orth, L=0.01, iternorm=True):
#     """ Find the orthogonal directions through eigendecomposition."""
#     orth = orth.cpu()
#     U = init_matrix
#     for i in range(len(orth)):
#         resid = U @ orth[i] @ U.t()
#         #print(resid)
#         D, Q = torch.symeig(resid, eigenvectors=True)
#         D = torch.relu(D-1e-4) + 1e-4 # numericial issues
#         Q = torch.diag(1/torch.sqrt(D)) @ Q.t()

#         #U = Q @ U
#         U = (Q*L) @ U + (1.0-L)*U
#         if iternorm: 
#             U = U / torch.norm(U, dim=0)
#     return U


# def build_pair_indices(n):
#     list1 = []
#     list2 = []
#     for i in range(n):
#         list1.append(i*torch.ones(i,dtype=torch.int64))
#         list2.append(torch.arange(i, dtype=torch.int64))
#     return torch.cat(list1), torch.cat(list2)


# def find_matrix_sphere_cluster(orthogonality, n_agg, sign_adapt=False, independent_samples=False, n_init=20):
#     n = len(orthogonality)
#     num_factors = orthogonality.shape[1]
#     orth_avg = orthogonality[:n].reshape(n//n_agg, n_agg, num_factors, num_factors)
#     orth_avg = orth_avg.mean(dim=1)
#     if not independent_samples:
#         a, b = build_pair_indices(n//n_agg)
#     else:
#         totlen = len(orth_avg)//2
#         a = slice(0,totlen)
#         b = slice(totlen, 2*totlen)
#     Hres, _ = find_B_with_chol(orth_avg[a], orth_avg[b])
#     Hresn = Hres.reshape(-1,num_factors)
#     if sign_adapt:
#         Hresn = Hresn * torch.sign(Hresn[:,0]).unsqueeze(1)
#     Hresn = Hresn / torch.norm(Hresn, dim=1, keepdim=True)
#     #skm = VonMisesFisherMixture(n_clusters=3*(1 if sign_adapt else 2), posterior_type='hard') #
#     skm = SphericalKMeans(num_factors*(1 if sign_adapt else 2), max_iter=1000, tol=1e-6, n_init=n_init)
#     skm.fit(Hresn)
#     Mdet = skm.cluster_centers_
    
#     plt.scatter(Hresn[:,1], Hresn[:,2], 2)
#     plt.scatter(Mdet[:,1], Mdet[:,2], 50, c="k")
    
#     # Match directions with their best counterpart.
#     if not sign_adapt:
#         best_opposites = match_cluster_centers(Mdet @ Mdet.T)
#         print("Matches", best_opposites)
#         # Select num_factors indices, that are not matched with each other...
#         selected_indices = np.arange(num_factors*2) < best_opposites
#         print("Selected", selected_indices)
#         print("Mirroring and refitting.")
#         print(selected_indices[skm.labels_].shape)
#         #Mdet = 0.5*(Mdet[selected_indices,:] - Mdet[selected_indices[best_opposites], :])
#         Hresn = Hresn*torch.tensor(np.sign(2*selected_indices[skm.labels_]-1).reshape(-1, 1), dtype=torch.float)
#         print("Refitting")
#         #skm = VonMisesFisherMixture(n_clusters=num_factors, posterior_type='hard') #
#         skm = SphericalKMeans(num_factors, max_iter=1000, tol=1e-6, n_init=n_init)
#         skm.fit(Hresn)
#         Mdet = skm.cluster_centers_
#     return torch.tensor(skm.cluster_centers_, dtype=torch.float)

def find_non_sing_submatrix(f):
    """ Return a non singular, square submatrix of M."""
    #print(f.shape)
    # 1st step: Find non singular submatrix.
    ind_set = []
    clm_count = 0
    for i in range(0, len(f)):
        ind_set.append(clm_count)
        while torch.svd(f[:,ind_set], compute_uv=False)[1][-1] < 1e-5: # linear independence check. Continue if independent.
            clm_count += 1
            ind_set[-1] = clm_count
        #print("Appending column", clm_count, "to set.")
        clm_count += 1
        if len(ind_set) == f.size(0):
            break
    #print(ind_set)
    if len(ind_set) < f.size(0):
        return None
    else:
        return f[:, ind_set]

# def find_matrix_sphere_cluster_disjoint(attrib, sign_adapt=False, n_init=20, use_idx = None):
#     """ attrib: matrix of shape [N, C, L] """
#     n = len(attrib)
#     num_factors = attrib.size(1)
#     #num_factors = orthogonality.shape[1]
#     #attr_avg = orthogonality[:n].reshape(n//n_agg, n_agg, num_factors, num_factors)
#     #attr_avg = attr_avg.mean(dim=1)
#     #if use_idx is None:
#     #    use_idx = torch.arange(num_factors)
    
#     # Exclude all non-invertible
#     Q_arr = []
#     for i in range(len(attrib)):
#         if use_idx is not None:
#             Q_arr.append(torch.inverse(attrib[i, :, use_idx]))
#         else:
#             Q_arr.append(torch.inverse(find_non_sing_submatrix(attrib[i])))

#     Q = torch.stack(Q_arr, dim=0)
#     print(Q.shape)
#     if len(Q) == 0:
#         print("All submatrices were singular")
#         return torch.eye(num_factors)

#     Hresn = Q.reshape(-1,num_factors)
#     if sign_adapt:
#         Hresn = Hresn * torch.sign(Hresn[:,0]).unsqueeze(1)
#     Hresn = Hresn / torch.norm(Hresn, dim=1, keepdim=True)
#     #skm = VonMisesFisherMixture(n_clusters=3*(1 if sign_adapt else 2), posterior_type='hard') #
#     skm = SphericalKMeans(num_factors*(1 if sign_adapt else 2), max_iter=1000, tol=1e-6, n_init=n_init)
#     skm.fit(Hresn)
#     Mdet = skm.cluster_centers_
    
#     plt.scatter(Hresn[:,1], Hresn[:,2], 2)
#     plt.scatter(Mdet[:,1], Mdet[:,2], 50, c="k")
    
#     # Match directions with their best counterpart.
#     if not sign_adapt:
#         best_opposites = match_cluster_centers(Mdet @ Mdet.T)
#         print("Matches", best_opposites)
#         # Select num_factors indices, that are not matched with each other...
#         selected_indices = np.arange(num_factors*2) < best_opposites
#         print("Selected", selected_indices)
#         print("Mirroring and refitting.")
#         print(selected_indices[skm.labels_].shape)
#         #Mdet = 0.5*(Mdet[selected_indices,:] - Mdet[selected_indices[best_opposites], :])
#         Hresn = Hresn*torch.tensor(np.sign(2*selected_indices[skm.labels_]-1).reshape(-1, 1), dtype=torch.float)
#         print("Refitting")
#         #skm = VonMisesFisherMixture(n_clusters=num_factors, posterior_type='hard') #
#         skm = SphericalKMeans(num_factors, max_iter=1000, tol=1e-6, n_init=n_init)
#         skm.fit(Hresn)
#         Mdet = skm.cluster_centers_
#     return torch.tensor(skm.cluster_centers_, dtype=torch.float)

# def match_cluster_centers(cosdistances_centers):
#     """ Perform a bipartite matching. """
#     workmat = cosdistances_centers.copy()
#     match_candidates = np.argsort(cosdistances_centers, axis=1)
#     #print("Overall best:", match_candidates[:, 0])
#     unmatched = np.ones(len(cosdistances_centers))
#     matches = np.zeros(len(cosdistances_centers), dtype=np.int)
#     while np.sum(unmatched) > 0:
#         # Find best unmatched cluster.
#         match_candidates = np.argsort(workmat, axis=1)
#         match_scores = np.min(workmat, axis=1)
#         best_match = np.argmin(match_scores, axis=0)
#         #print(match_scores)
#         if best_match == match_candidates[best_match, 0]:
#             raise ValueError("Invariant matching matrix")
#         print("Maching:", best_match, " and ", match_candidates[best_match, 0])
#         partner = match_candidates[best_match, 0]
#         workmat[best_match, :] = 1
#         workmat[partner, :] = 1
#         workmat[:, best_match] = 1
#         workmat[:, partner] = 1
#         matches[partner] = best_match
#         matches[best_match] = partner
#         unmatched[partner] = 0
#         unmatched[best_match] = 0
#     return matches

# Some simple metrics to measure linear disentanglement.

def compute_change_matrix(found_directions, ground_truth_dirs, inverse=True, d=0):
    """ Compute the effect of found directions. This refers to the change when going in the discovered direction by one unit.
        Ground truth dirs: [Z (latent dim), N (num directions)]
        found_directions: [Z (latent dim), M (num directions)]
        Equation:
        z' = FDz
        z = (FD)^-1z'
        where F is the the Found directions, D is the ground truth directions.
    """ 
    if inverse:
        ret =  torch.abs(torch.inverse(found_directions.matmul(ground_truth_dirs)))
    else:
        ret =  torch.abs(found_directions.matmul(ground_truth_dirs))
    return ret / torch.norm(ret, dim=d, keepdim=True)


def change_matrix_disentanglement(found_directions, ground_truth_dirs, two_way=False, inverse=True):
    """ Our directional disentanglement metric. 
        The score is computed by summing up the differences between the most relevant 
        change in one dimension and the second most relevant for each latent dimension.
        If the dimension is perfectly disentangled, the score for this latent dimension is 1.
        The total score is in the range [0, num_latent directions] where 0 indicates no disentanglement an num_latent directions
        indicates perfect disentanglement.
    """
    sorted_effects = torch.sort(compute_change_matrix(found_directions, ground_truth_dirs, inverse=inverse, d=0), dim=0, descending=True)[0]
    disentanglement = (sorted_effects[0,:] - sorted_effects[1,:])
    if two_way:
        sorted_effects = torch.sort(compute_change_matrix(found_directions, ground_truth_dirs, inverse=inverse, d=1), dim=1, descending=True)[0]
        disentanglement += (sorted_effects[:,0] - sorted_effects[:,1])
        disentanglement = disentanglement*0.5
    return torch.sum(disentanglement)


def regression_matrix_disentanglement(model, concept_fn, dirs, sample_epochs = 50):
    """ This function measures disentanglement by attempting to regress the concepts using
        only the scores in the discovered direction. This works well, when concepts are binary.
    """
    ret_mat = gt_get_binary_score_matrix(model, concept_fn, dirs, sample_epochs = sample_epochs)
    dirmat = 2*torch.abs(ret_mat-0.5) # 0.5 means random accuracy
    sorted_effects = torch.sort(dirmat, dim=0, descending=True)[0]
    disentanglement =(sorted_effects[0,:] - sorted_effects[1,:])
    return torch.sum(disentanglement)


# def compute_components_svd(net, testloader, mfv, num_batches=20):
#     J_batchlist = []
#     inp_shape = 0
#     for i, data in enumerate(testloader):
#         if i == num_batches:
#             break
#         _, x, _ = tuple([x.to(net.device) for x in data])
#         J = integrated_gradients(net, x, mfv)
#         inp_shape = J.shape
#         J = J.reshape(J.size(0)*J.size(1), -1)
        
#         print(inp_shape)
#         J_batchlist.append(J)
#     avg_J = torch.cat(J_batchlist, dim=0)
#     print(avg_J.shape)
#     mean_J = avg_J.reshape(-1, inp_shape[1], avg_J.size(-1)).mean(dim=0)
#     print(mean_J.shape)
#     U, S, V = torch.svd(avg_J.t(), some=True)
#     U_use = U[:, :inp_shape[1]]
#     p_inv = V[:, :inp_shape[1]].reshape(-1, inp_shape[1], inp_shape[1])
#     print(p_inv.shape)
#     #p_inv = torch.pinverse(U_use).matmul(mean_J.t())
#     return U, S, V, p_inv.mean(dim=0)