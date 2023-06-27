### Some analysis functions for the post hoc analysis of models.
from numpy import common_type
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FastICA
from common.data_loader import DisentanglementLibDataset, ResamplingDataLoader
import numpy as np
from tqdm import tqdm
from copy import deepcopy


# def gt_concept_directions_by_regression(data_loader, model, sample_epochs):
#     """ Find the ground truth embedding directions for a model. 
#         To find these direction, Logistic Regression between different
#         values for the concept is employed.
#     """
#     dl_iter = iter(data_loader)
#     example_batch = next(dl_iter)
#     num_gt_factors = example_batch[0].size(1)
#     print(f"Dataset has {num_gt_factors} factors.")
#     ret_mat = torch.zeros(model.z_dim, num_gt_factors)
#     for f in range(num_gt_factors):
#         # Sample a test set. 
#         common_values = torch.unique(example_batch[0][:,f])
#         common_values = common_values[torch.randperm(len(common_values))]
#         print(f"Unique values of factor {f}: {str(common_values)}")
#         encodings = []
#         labels = []
#         # Regress value 1 vs value 2.
#         encodings = {i : [] for i in range(len(common_values))}
#         dl_iter = iter(data_loader)
#         for k in range(min(sample_epochs, len(data_loader)-1)):
#             new_batch = next(dl_iter)
#             inp_batch = deepcopy(new_batch[1])
#             gt_fact = deepcopy(new_batch[0])[:,f]

#             for g in range(len(common_values)):
#                 inp = inp_batch[gt_fact==common_values[g].item()]
#                 if len(inp):
#                     encs = model.encode_deterministic(images = inp.to(model.device))
#                     encodings[g].append(encs.detach().cpu())
            
#         for g in range(len(common_values)):
#             encodings[g] = torch.cat(encodings[g], dim=0)


#         # print([e.shape for e in encodings.values()])       
#         # We now have a training dataset. Now regress the directions.
#         ret_acc = torch.zeros(model.z_dim)
#         for i in range(len(common_values)):
#             for j in range(i):
#                 #print(i,j)
#                 in_labels = torch.cat((torch.zeros(len(encodings[i])), torch.ones(len(encodings[j]))), dim=0)
#                 in_features = torch.cat((encodings[i], encodings[j]), dim=0)
#                 # Perform regression.
#                 lrmodel = LogisticRegression(solver='lbfgs')
#                 lrmodel.fit(in_features, in_labels)
#                 coeff = torch.from_numpy(lrmodel.coef_).float().flatten()
#                 if torch.sum(ret_acc*coeff) > 0:
#                     ret_acc += coeff/torch.norm(coeff)
#                 else:
#                     ret_acc -= coeff/torch.norm(coeff)
#         ret_mat[:,f] = ret_acc
#     return ret_mat


# def gt_concept_directions_by_variance(dataset: DisentanglementLibDataset, model, n_samples = 20, denormalize = False, factor_means=None):
#     """ 
#         Return the ground truth directions for each concept.
#         I this method, all but one ground truth factor is kept constant. The images
#         are encoded then. On the encodings a truncated PCA is run, i.e., only the the first
#         principal direction with highest variance is returned.
#         returns a tuple with the direction and the factor values that are used as a basis of the transformation.
#     """
#     factor_values = dataset.dataset.factors_num_values
#     if factor_means is None:
#         print(dataset.dataset.factors_num_values)
#         #factor_means = torch.tensor([5, 5, 5, 4, 2, 7])
#         factor_means = np.array([int(factor_value/2) for factor_value in factor_values])
#         #data_mean = dataset.sample_observations_from_factors(factor_means.reshape(1,-1))
#         #mean_tensor = torch.from_numpy(np.moveaxis(data_mean, 3, 1), ).type(torch.FloatTensor)
#         #encs_mean = model.encode_deterministic(images = mean_tensor.to(model.device))
#         #print(encs_mean)
#     ret_mat = torch.zeros(model.z_dim, len(factor_values))
#     ret_mags = torch.zeros(len(factor_values))
#     for i in range(len(factor_values)):
#         # sample random factor values.
#         r_samples = torch.randint(0, factor_values[i], (n_samples,))
#         #r_samples[r_samples==factor_means[i]] = factor_values[i]-1 # set means to last value.
#         #print("Samples:", r_samples)
#         cng_factor = np.tile(factor_means.copy().reshape(1,-1), (n_samples, 1))
#         cng_factor[:, i] = r_samples.numpy()
#         #print(cng_factor)
#         data_batch = dataset.sample_observations_from_factors(cng_factor)
#         #print(data_batch.shape)
#         batch_tensor = torch.from_numpy(np.moveaxis(data_batch, 3, 1), ).type(torch.FloatTensor)
#         with torch.no_grad():
#             encs = model.encode_deterministic(images = batch_tensor.to(model.device)).detach()
#         #encs = 0.1*encs + torch.mean(encs, dim=0, keepdim=True)
#         #encs[:, i] += 5*torch.randn(len(encs), device=model.device)
#         #print(encs)
#         # Find direction of highest variance in encodings.
#         ediff = (encs-torch.mean(encs, dim=0, keepdim=True))
#         #print(ediff)
#         #ediff /= torch.norm(ediff, dim=1, keepdim=True)
#         cov_mat = ediff.t().matmul(ediff)
#         #import matplotlib.pyplot as plt
#         #plt.matshow(cov_mat.cpu())
#         # print(cov_mat.shape)
#         #print(cov_mat)
#         vals, S = torch.symeig(cov_mat, eigenvectors=True)
#         print("Eigenvalues: ", vals.cpu().numpy())
#         ret_mat[:,i] = S[:, -1]
#         ret_mags[i] = vals[-1]
#     if denormalize:
#         #print(ret_mags)
#         mags = torch.sqrt(ret_mags)
#         mags_norms = len(mags)*mags/torch.sum(mags)
#         #print(mags_norms)
#         ret_mat *= mags_norms.reshape(1,-1)
#     return ret_mat, factor_means


# def gt_regres_binary_directions(model, concept_fn, sample_epochs = 100, uncorr_loader = True):
#     """ Regress in latent space to find the direction of a certain variation.
#         d: Latent encodings
#         c: Binary concepts 
#     """
#     enc_list = []
#     fact_list = [] 
#     gt_fact_list = []
#     if uncorr_loader:
#         data_loader = _get_uncorrelated_dl(model.data_loader, model.filter_fn)
#     else:
#         data_loader = model.data_loader

#     gt_fact_list, enc_list, _, fact_list = \
#         _sample_eval_dataset(model, data_loader, sample_epochs=sample_epochs, binary_concept_fn = concept_fn)
#     print(enc_list.shape, fact_list.shape)
#     # Calculate min-max normalized label_direction.
#     res = torch.zeros(enc_list.size(1), fact_list.size(1))
#     for j in range(fact_list.size(1)):
#         rfc1 = LogisticRegression()
#         rfc1.fit(enc_list.numpy(), fact_list[:, j].numpy())
#         print(f"Score {j}: {rfc1.score(enc_list.numpy(), fact_list[:, j].numpy())}")
#         res[:,j] = torch.tensor(rfc1.coef_)
#         res[:,j] /= res[:,j]. norm()
#     return res


def gt_get_binary_score_matrix(model, concept_fn, dirs, sample_epochs = 100, uncorr_loader = True):
    """ 
        Regress the concepts for each of the found directions.
        dirs: [Z, N]-matrix with one of the directions in each columns
        DEPRECATED -> use common.metrics.sap_score instead
    """
    enc_list = []
    fact_list = [] 
    gt_fact_list = []
    if uncorr_loader:
        data_loader = _get_uncorrelated_dl(model.data_loader, model.filter_fn)
    else:
        data_loader = model.data_loader

    gt_fact_list, enc_list, _, fact_list = \
        _sample_eval_dataset(model, data_loader, sample_epochs=sample_epochs, binary_concept_fn = concept_fn)


    # Calculate min-max normalized label_direction.
    res = torch.zeros(fact_list.size(1), fact_list.size(1))
    for j in range(fact_list.size(1)): # project on direction.
        inputs = torch.sum(dirs[:,j].reshape(1,-1)*enc_list, dim=1, keepdim=True)
        #print(inputs.shape)
        for k in range(fact_list.size(1)): # ground truth concepts.
            rfc1 = LogisticRegression(solver='liblinear')
            rfc1.fit(inputs.numpy(), fact_list[:, k].numpy())
            res[j, k] = rfc1.score(inputs.numpy(), fact_list[:, k].numpy())
    return res

# def get_eigenspace_directions(model, data_loader, num_dirs, iters=50, denormalize=True):
#     """ Return the n_dirs principal directions of the latent space,
#         denormalized by their variance, such that the variance when multiplied with these
#         directions is approximately unit.
#         return [Z_dim, num_dirs] matrix with num dir column vectors.
#     """
#     #num_dirs = 4
#     eigvects, eigvals = compute_pca(model, data_loader, iters)
#     if denormalize:
#         init_matrix = eigvects[:,:num_dirs]*eigvals[:num_dirs].unsqueeze(0).pow(-0.5)
#     else:
#         init_matrix = eigvects[:,:num_dirs]
#     return init_matrix

def get_ica_directions(model, data_loader, num_dirs, iters=50):
    """ Compute principal direction using Independent Component Analysis (ICA). """
    _, z_batch_list, _ = _sample_eval_dataset(model, data_loader, sample_epochs=iters)
    fastica = FastICA(n_components=num_dirs, random_state=0)
    fastica.fit(z_batch_list)
    return torch.tensor(fastica.components_, dtype=torch.float).t()


def compute_pca(model, data_loader, iters=50, fit_sklearn=False):
    """ Calculate the principal components of the 
        latent space using Principal Component Analysis (PCA). Return the Directions
        along with the corresponding eigenvalues, ordered descendingly.
    """
    _, z_batch_list, _ = _sample_eval_dataset(model, data_loader, sample_epochs=iters)

    # Either use sklearn PCA
    if fit_sklearn:
        pca = PCA(n_components=z_batch_list.size(1))
        pca.fit(z_batch_list)
        eivect_pca = pca.components_
        sing_values = pca.singular_values_
        return torch.tensor(eivect_pca, dtype=torch.float).t(), torch.tensor(sing_values, dtype=torch.float)
    else: # use a full torch implementation (the results are equivalent)
        z_batch_list -= z_batch_list.mean(dim=0, keepdim=True)
        cov_mat = z_batch_list.t().matmul(z_batch_list)/len(z_batch_list)
        vals, S = torch.symeig(cov_mat, eigenvectors=True)
        # Eigenvalues are ordered ascendingly -> change to descending
        return torch.flip(S, (1,)).cpu(), torch.flip(vals, (0,)).cpu()

def _get_uncorrelated_dl(data_loader, filter_fn = None):
    """ Create a new data loader object on the same data set
        that does not return correlated data. This is required for an unbiased evaluation 
        of the disentanglement scores.
        This only works if the correlated data_loader is an instance of class ResamplingDataLoader (common.data_loader)
        :params data_loader: The data_loader object.
        :params filter_fn: Filter_fn that should also be applied to the new data loader.
    """
    if type(data_loader) == ResamplingDataLoader:
        if filter_fn is None:
            dl_ret = data_loader.my_dl
        else:
            dl_ret = ResamplingDataLoader(data_loader.my_dl, data_loader.oversampling_factor, output_dens=filter_fn)
        return dl_ret
    else: # A standard data loader -> No changes.
        return data_loader

def _sample_eval_dataset(model, data_loader, sample_epochs=50, binary_concept_fn = None):
    """ Sample a subset of the data for testing. 
        The total size sampled will be n_iters*data_loader.batch_size.
        Return a tuple of
        (input_factors, encodings, labels, binary factors)
    """
    gt_factor_list = [] # ground truth factors
    z_encodings_list = [] # latent codes
    bin_factor_list = [] # binary concept factors (only if binary_concept_fn is not None)
    lables_list = [] # labels
    dl_iter = iter(data_loader)
    for k in tqdm(range(min(sample_epochs, len(data_loader)))):
        new_batch = next(dl_iter)
        inp_batch = deepcopy(new_batch[1])
        gt_fact = deepcopy(new_batch[0])
        gt_factor_list.append(gt_fact)
        lables_list.append(deepcopy(new_batch[2]))
        with torch.no_grad():
            encs = model.encode_deterministic(images = inp_batch.to(model.device)).detach().cpu()
        z_encodings_list.append(encs)
        if binary_concept_fn is not None:
            bin_factor_list.append(binary_concept_fn(gt_fact))
        
    z_encodings_list = torch.cat(z_encodings_list, dim=0)
    gt_factor_list = torch.cat(gt_factor_list, dim=0)
    lables_list = torch.cat(lables_list, dim=0)
    if binary_concept_fn is not None:
        bin_factor_list = torch.cat(bin_factor_list, dim=0)
        return gt_factor_list, z_encodings_list, lables_list, bin_factor_list
    else:
        return gt_factor_list, z_encodings_list, lables_list