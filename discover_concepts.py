## script for post-hoc concept discovery
from sklearn.neighbors import VALID_METRICS
import sys
import torch
from torch.optim import Adam, RMSprop, SGD
from tqdm import tqdm
import logging
import time

import models
import common.attributions
from common.utils import setup_logging
from common.analysis import _get_uncorrelated_dl, gt_concept_directions_by_variance
from common.analysis import gt_get_binary_score_matrix, get_eigenspace_directions, get_ica_directions
from common.posthoc_orthogonalizer import disjoint_orthogonality_loss
from common.custom_logger import CustomLogger
from common.arguments import get_args, get_args_search
from aicrowd.aicrowd_utils import evaluate_disentanglement_metric_with_linear_transformation

def load_model(checkpoint_file, batch_size, dset_dir, alg="BetaVAE", corr=0.0, 
        encoder_name="SimpleGaussianConv64", decoder_name="SimpleConv64", filter_fn=None, dataset = "shapes3d",
        n_cor=0, image_size=64, overwrite_cor_list=False, corr_feature1=0, corr_feature2=1, z_dim=6, oversampling=3 , log_dir="."):
    """ Load a pretrained model for posthoc orthogonalization from an existing checkpoint (created by running main.py)
        Parameters:
            checkpoint_file: Checkpoint file path to load
            batch_size: batch size to use during post-hoc discovery with SGD
            dset_dir: path to the datasets
            alg: Algorithm to use, either "BetaVAE" or "Discriminative"
            corr: Correlation strengh introduced in the dataloader
            encoder_name / decoder_name: Encoder and Decoder Architectures
            filter_fn: Filter function that may limit the data support (pass None, if the entire dataset should be used)
            dataset: name of the dataset; see common.constants.DATASETS for options.
            n_cor: Number of correlated factors
            image_size: input image size (images are always square with one length corresponding to image_size pixels)
            overwrite_cor_list: 
            corr_feature1: index of first correlated feature
            corr_feature2: index of second correlated feature
            z_dim: Number of latent dimensions
            oversampling: Oversampling factor (i.e., how many batches of uncorrelated data are sampled to produce one batch of correlated data)
            log_dir: where to store the log files and results
    """
    print("Loading model... ")
    if alg == "BetaVAE":
        args_load = get_args([f'--name=temp', '--alg=BetaVAE', f'--dset_dir={dset_dir}', 
                    f'--dset_name={dataset}', f'--encoder={encoder_name}', f'--decoder={decoder_name}',
                    f'--z_dim={z_dim}', '--w_kld=1', '--use_wandb=false', '--sigma_corr', f'{corr}', '--max_epoch', '5',
                    '--max_iter', '2000', '--oversampling_factor', str(oversampling),
                    '--ckpt_load', checkpoint_file, '--batch_size', str(batch_size), '--evaluation_metric', 'factor_vae_metric',
                    f'--filter_fn={filter_fn}', f'--n_cor={n_cor}', f'--overwrite_cor_list={overwrite_cor_list}',
                    f'--corr_feature1={corr_feature1}', f'--corr_feature2={corr_feature2}', f'--ckpt_dir={log_dir}', 
                    f'--train_output_dir={log_dir}', f'--test_output_dir={log_dir}', f'--image_size={image_size}'])

    elif alg == "Discriminative":
        args_load = get_args([f'--name=temp', '--alg=Discriminative', f'--dset_dir={dset_dir}', 
                    f'--dset_name={dataset}', f'--z_dim={z_dim}', '--w_kld=1', '--use_wandb=false', "--classifier=GAPClassifier", '--sigma_corr', f'{corr}',
                    '--corr_feature1', '0', '--corr_feature2', '1', 
                    '--max_iter', '2000', '--oversampling_factor', '5', "--labeling_fn", "shapes3d_binary_labelfunction_eight",
                    '--ckpt_load', checkpoint_file, '--batch_size', str(batch_size), '--evaluation_metric', 'factor_vae_metric', 
                    f'--ckpt_dir={log_dir}',  f'--train_output_dir={log_dir}',  f'--test_output_dir={log_dir}'])
    print(args_load.dset_dir)
    model_cl = getattr(models, args_load.alg)
    model = model_cl(args_load)
    model.load_checkpoint(args_load.ckpt_load, load_optim=args_load.ckpt_load_optim)
    return model


def train_loop(args, net, A_init, attribution_fn, loss_fn, logger, n_iter, ground_truth_dirs=None, optimize=True):
    """ Main look to recover the matrix M such that the directions fulfill the IMA/DMA criterion. 
        Parameters:
            args: Command line arguments
            net: The trained encoder.
            A_init: Intial matrix M at the start of the optimization
            attribution_fn: Function to compute the gradients. Can be used to implement variations
                of the gradient such as integrated or smoothed grad.
            loss_fn: Function that computes the loss
            logger: Custom logger object to log the optimization progress
            n_iter: Number of iterations to perform
            ground_truth_dirs: Ground truth directions if known (required to compute some alignment metrics)
            optimize: True, if A should be optimized, otherwise only the loss and the metrics are computed and logged.
        Returns the matrix with the discovered directions and the average loss
    """

    if optimize:
        A = A_init.clone().to(net.device)
        A.requires_grad_(True)
        
        if args.optimizer == "adam":
            opt = Adam((A,), lr=args.learning_rate)
        elif args.optimizer == "rmsprop":
            opt = RMSprop((A,), lr=args.learning_rate)
        elif args.optimizer == "sgd":
            opt = SGD((A,), lr=args.learning_rate)
        else:
            raise ValueError("Unknown optimizer.")
    else:
        A = A_init.to(net.device)

    running_loss = 0.0
    cum_loss = 0.0
    testloader = net.data_loader
    with tqdm(total=n_iter, position=0, leave=True) as pbar:
        epoch_sz = len(testloader)
        for epoch in range(n_iter//epoch_sz + 1):
            for i, data in enumerate(testloader):
                _, x, _ = tuple([x.to(net.device) for x in data])
                J = attribution_fn(net, x).detach().contiguous()
                J_orgshape = list(J.shape[2:])
                J = J.view(J.size(0), J.size(1), -1) # [B, latent_dim, input_dim]
                disentangledJ = torch.matmul(A.unsqueeze(0), J)
                disentangledJ =  disentangledJ.reshape(J.size(0), A.size(0), *J_orgshape)
                loss = torch.mean(loss_fn(disentangledJ)[1])
                if optimize:
                    loss.backward()
                    opt.step()
                running_loss += loss.detach().item()
                cum_loss += loss.detach().item()
                pbar.update(1)
                if epoch*epoch_sz + i >= n_iter:
                    break   
                if (epoch*epoch_sz + i) % args.evaliter == (args.evaliter -1) and optimize:
                    print("Running loss: ", running_loss/args.evaliter)
                    eval_results = evaluate_disentanglement_metric_with_linear_transformation(net, A,
                            args.evaluation_metric, ground_truth_dirs=ground_truth_dirs)
                    eval_results.update({"iter": epoch*epoch_sz + i, "oloss": running_loss/args.evaliter})
                    logger.update(eval_results)
                    running_loss = 0.0
    return A.detach(), cum_loss/n_iter


def setup_functions(args, model):
    """ 
        Setup functions for the optimization in "train_loop" according to the parameters passed in args.
        Parameters:
            args: the command line arguments
            model: the model to be used
        Returns loss_fn, attribution_fn 
    """
    if args.attribution == "grad":
        attribution_fn = common.attributions.input_jacobian
    elif args.attribution == "ig":
        print("Calculating mean feature for ig...")
        mfv = common.attributions.calc_mean_feature_value(model.data_loader)
        attribution_fn = lambda net, x: common.attributions.integrated_gradients(net, x, baseline = mfv)
    elif args.attribution == "sg":
        attribution_fn = common.attributions.smoothgrad_gradients
    elif args.attribution == "gen":
        attribution_fn = common.attributions.decoder_change
    else:
        raise ValueError("Illegal attribution.")

    loss_fn = lambda x: disjoint_orthogonality_loss(x, normalize=args.norm, disjoint=args.disjoint, losstype=args.losstype)

    return attribution_fn, loss_fn


def main(args):
    # Store the most important parameters in a dict
    params_dict = {"attrib": args.attribution}
    params_dict["disj"] = args.disjoint
    params_dict["norm"] = args.norm
    params_dict["optim"] = args.optimizer
    params_dict["lr"] = args.learning_rate
    params_dict["batchsize"] = args.batchsize
    params_dict["loss"] = args.losstype

    logger = CustomLogger(args.logdir, params_dict, 2, exit_on_exist=not args.recompute_baselines)
    logger.add_single_value("checkpoint_path", args.checkpoint)
    model = load_model(args.checkpoint, args.batchsize, args.dsetdir, args.alg, args.sigma_corr, args.encoder, args.decoder,
                       filter_fn=args.filter_fn, n_cor=args.n_cor, overwrite_cor_list=args.overwrite_cor_list, image_size = args.image_size,
                       corr_feature1=args.corr_feature1, corr_feature2=args.corr_feature2, oversampling=args.oversampling_factor,
                       log_dir = args.logdir, dataset=args.dset, z_dim=args.z_dim)

    attribution_fn, loss_fn = setup_functions(args, model)

    # First run the baselines and compute the metrics.
    init_matrix = torch.eye(args.num_directions, model.z_dim) # or randn
    print("Init matrix shape:", init_matrix.shape)

    gt_dirs = None

    # 1: Baseline: The Unit Directions.
    eval_unit = evaluate_disentanglement_metric_with_linear_transformation(model, init_matrix, args.evaluation_metric, ground_truth_dirs=gt_dirs)
    logger.add_single_value("baseline_unit", eval_unit)

    if not args.skip_baselines:
        # 2. Baseline PCA
        start_pca = time.time()
        eigendirs = get_eigenspace_directions(model, model.data_loader, args.num_directions, iters=args.eval_batches)
        end_pca = time.time()
        logger.add_single_value("pca_time", end_pca-start_pca)
        eval_pca = evaluate_disentanglement_metric_with_linear_transformation(model, eigendirs.t(), args.evaluation_metric, ground_truth_dirs=gt_dirs)
        logger.add_single_value("baseline_pca", eval_pca)
        logger.add_single_value("eigendirs", eigendirs.cpu().numpy().tolist())

        # 3. Baseline ICA
        start_ica = time.time()
        icavects = get_ica_directions(model, model.data_loader, args.num_directions, args.eval_batches)
        end_ica = time.time()
        logger.add_single_value("ica_time", end_ica-start_ica)
        eval_ica = evaluate_disentanglement_metric_with_linear_transformation(model, icavects.t(), args.evaluation_metric, ground_truth_dirs=gt_dirs)
        logger.add_single_value("baseline_ica", eval_ica)
        logger.add_single_value("icavects", icavects.cpu().numpy().tolist())

    if not args.recompute_baselines:
        start_ours = time.time()
        res_dirs, _ = train_loop(args, model, init_matrix, attribution_fn, loss_fn, logger, n_iter = args.max_iter, ground_truth_dirs=gt_dirs)
        end_ours = time.time()
        logger.add_single_value("ours_time", end_ours-start_ours) 
        # Note that for a fair time comparison, no evaluation should be conducted in train_loop for example by setting args.eval_iter to a large value
        logger.add_single_value("final_dirs", res_dirs.cpu().numpy().tolist())
    logger.close()

if __name__ == "__main__":
    args = get_args_search(sys.argv[1:])
    main(args)
