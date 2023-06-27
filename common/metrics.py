# Our own implementation of some disentanglement metrics.
from common.analysis import _get_uncorrelated_dl
import torch
from sklearn.linear_model import LogisticRegression

def sap_score(model, concept_fn, dirs, sample_epochs = 100, uncorr_loader = True):
    """ Implement the SAP (Seperate Attribute Predictability) score by 
        Kumar et al.: Variational inference on disentangled latent 
        concepts from unlabeled observations (2017) 
    """
    enc_list = []
    fact_list = [] 
    gt_fact_list = []
    if uncorr_loader:
        dl_iter = iter(_get_uncorrelated_dl(model.data_loader, model.filter_fn))
    else:
        dl_iter = iter(model.data_loader)
    for k in range(sample_epochs):
        new_batch = next(dl_iter)
        inp_batch = new_batch[1]
        gt_fact = new_batch[0]
        encs = model.encode_deterministic(images = inp_batch.to(model.device)).detach().cpu()
        enc_list.append(encs)
        if concept_fn is not None:
            fact_list.append(concept_fn(gt_fact))
        else:
            fact_list.append(gt_fact)

        gt_fact_list.append(gt_fact)
    enc_list = torch.cat(enc_list, dim=0)
    fact_list = torch.cat(fact_list, dim=0)
    #print(enc_list.shape, fact_list.shape)

    # Calculate the Informativeness Matrix I
    # The ij entry is the score when predicting the j ground truth factor from the ith latent code.
    Imatrix = torch.zeros(dirs.size(1), fact_list.size(1))
    for i in range(dirs.size(1)): # project on direction.
        inputs = torch.sum(dirs[:,i].reshape(1,-1)*enc_list, dim=1, keepdim=True)
        #print(inputs.shape)
        for j in range(fact_list.size(1)): # ground truth concepts.
            # Use multinomial for multiclass, apply (almost) no regularization
            rfc1 = LogisticRegression(multi_class= "multinomial", solver="lbfgs", C=1e4)
            rfc1.fit(inputs.numpy(), fact_list[:, j].numpy())
            Imatrix[i, j] = rfc1.score(inputs.numpy(), fact_list[:, j].numpy())

    # print(Imatrix)
    # Compute the SAP score.
    sorted_effects = torch.sort(Imatrix, dim=0, descending=True)[0]
    disentanglement = (sorted_effects[0, :] - sorted_effects[1, :])
    return torch.sum(disentanglement)