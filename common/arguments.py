import argparse
import os
import logging

from common import constants as c
from common.utils import str2bool, StoreDictKeyPair


def update_args(args):
    args.ckpt_load_iternum = False
    args.use_wandb = False
    args.file_save = True
    args.gif_save = True
    return args


def get_args(sys_args):
    """ Argument parsing for the training routines. """
    parser = argparse.ArgumentParser(description='disentanglement-pytorch')

    # NeurIPS2019 AICrowd Challenge
    parser.add_argument('--aicrowd_challenge', default=False, type=str2bool, help='Run is an AICrowd submission')
    parser.add_argument('--evaluation_metric', default=None, type=str, choices=c.EVALUATION_METRICS, nargs='+',
                        help='Metric to evaluate the model during training')

    # name
    parser.add_argument('--alg', type=str, help='the disentanglement algorithm', choices=c.ALGS)
    parser.add_argument('--controlled_capacity_increase', help='to use controlled capacity increase', default=False)
    parser.add_argument('--loss_terms', help='loss terms to be incldued in the objective', nargs='*',
                        default=list(), choices=c.LOSS_TERMS)
    parser.add_argument('--name', default='unknown_experiment', type=str, help='name of the experiment')

    # Neural architectures
    parser.add_argument('--encoder', type=str, nargs='+', choices=c.ENCODERS,
                        help='name of the encoder network')
    parser.add_argument('--decoder', type=str, nargs='+', choices=c.DECODERS,
                        help='name of the decoder network')
    parser.add_argument('--label_tiler', type=str, nargs='*', choices=c.TILERS,
                        help='the tile network used to convert one hot labels to 2D channels')
    parser.add_argument('--discriminator', type=str, nargs='*', choices=c.DISCRIMINATORS,
                        help='the discriminator network')
    parser.add_argument('--classifier', type=str, nargs='*', choices=c.CLASSIFIERS,
                        help='the classifier network used in the Discriminative Algorithm')
    # Test or train
    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_iter', default=3e7, type=float, help='maximum training iteration')
    parser.add_argument('--max_epoch', default=3e7, type=float, help='maximum training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_disc_layers', default=5, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_disc_layers', default=1000, type=int, help='size of fc layers in discriminators')

    # latent encoding
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, nargs='*',
                        help='Labels (indices or names) to include in latent encoding.')
    parser.add_argument('--l_dim', default=0, type=str, help='size of the encoded w space (for each label)')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')

    # Neural architectures hyper-parameters
    parser.add_argument('--num_layer_disc', default=6, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_layer_disc', default=1000, type=int, help='size of fc layers in discriminators')

    # Loss weights and parameters [Common]
    parser.add_argument('--w_recon', default=1.0, type=float, help='reconstruction loss weight')
    parser.add_argument('--w_kld', default=1.0, type=float, help='main KLD loss weight (e.g. in BetaVAE)')

    # Loss weights and parameters for [CapacityVAE]
    parser.add_argument('--max_c', default=25.0, type=float, help='maximum value of control parameter in CapacityVAE')
    parser.add_argument('--iterations_c', default=100000, type=int, help='how many iterations to reach max_c')

    # Loss weights and parameters for [FactorVAE & BetaTCVAE]
    parser.add_argument('--w_tc', default=1.0, type=float,
                        help='total correlation loss weight (e.g. in FactorVAE and BetaTCVAE)')

    # Loss weights and parameters for [InfoVAE]
    parser.add_argument('--w_infovae', default=1.0, type=float,
                        help='mmd loss weight (e.g. in InfoVAE)')

    # Loss weights and parameters for [DIPVAE I & II]
    parser.add_argument('--w_dipvae', default=1.0, type=float,
                        help='covariance regularizer loss weight (e.g. in DIPVAE I and II)')

    # Loss weights and parameters for [IFCVAE]
    parser.add_argument('--w_le', default=1.0, type=float, help='label encoding loss weight (e.g. in IFCVAE)')
    parser.add_argument('--w_aux', default=1.0, type=float, help='auxiliary discriminator loss weight (e.g. in IFCVAE)')

    # Hyperparameters for [DIP-VAE]
    parser.add_argument('--lambda_d_factor', default=10.0, type=float,
                        help='Hyperparameter for diagonal values of covariance matrix')
    parser.add_argument('--lambda_od', default=1.0, type=float,
                        help='Hyperparameter for off diagonal values of covariance matrix.')

    # Dataset
    parser.add_argument('--dset_dir', default=os.getenv('DISENTANGLEMENT_LIB_DATA', './data'),
                        type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, choices=c.DATASETS, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers for the data loader')
    parser.add_argument('--pin_memory', default=False, type=str2bool,
                        help='pin_memory flag of data loader. Check this blogpost for details:'
                             'https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/')
    parser.add_argument('--device', default="cuda:0", type=str, help='which device to use for model training')

    # Logging and visualization
    parser.add_argument('--train_output_dir', default='train_outputs', type=str, help='output directory')
    parser.add_argument('--test_output_dir', default='test_outputs', type=str, help='test output directory')
    parser.add_argument('--file_save', default=True, type=str2bool, help='whether to save generated images to file')
    parser.add_argument('--gif_save', default=False, type=str2bool, help='whether to save generated GIFs to file')
    parser.add_argument('--use_wandb', default=False, type=str2bool, help='use wandb for logging')
    parser.add_argument('--wandb_resume_id', default=None, type=str, help='resume previous wandb run with id')
    parser.add_argument('--traverse_spacing', default=0.2, type=float, help='spacing to traverse latents')
    parser.add_argument('--traverse_min', default=-2, type=float, help='min limit to traverse latents')
    parser.add_argument('--traverse_max', default=+2, type=float, help='max limit to traverse latents')
    parser.add_argument('--traverse_z', default=False, type=str2bool, help='whether to traverse the z space')
    parser.add_argument('--traverse_l', default=False, type=str2bool, help='whether to traverse the l space')
    parser.add_argument('--traverse_c', default=False, type=str2bool, help='whether to traverse the condition')
    parser.add_argument('--verbose', default=20, type=int, help='verbosity level')

    # Save/Load checkpoint
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_load_iternum', default=True, type=str2bool, help='start global iteration from ckpt')
    parser.add_argument('--ckpt_load_optim', default=True, type=str2bool, help='load the optimizer state')

    # Iterations [default for all is equal to 1 epoch]
    parser.add_argument('--treat_iter_as_epoch', default=False, type=bool, help='treat all iter arguments as epochs')
    parser.add_argument('--ckpt_save_iter', default=None, type=int, help='iters to save checkpoint [default: 1 epoch]')
    parser.add_argument('--evaluate_iter', default=None, type=int, help='iters to evaluate [default: 1 epoch]')
    parser.add_argument('--float_iter', default=None, type=int, help='iters to aggregate float logs [default: 1 epoch]')
    parser.add_argument('--print_iter', default=None, type=int, help='iters to print float values [default: 1 epoch]')
    parser.add_argument('--all_iter', default=None, type=int, help='use same iteration for all [default: 1 epoch]')
    parser.add_argument('--recon_iter', default=None, type=int, help='iters to reconstruct image [default: 1 epoch]')
    parser.add_argument('--traverse_iter', default=None, type=int, help='iters to visualize latents [default: 1 epoch]')
    parser.add_argument('--schedulers_iter', default=None, type=int, help='iters to apply scheduler [default: 1 epoch]')

    # Schedulers
    parser.add_argument('--lr_scheduler', default=None, type=str, choices=c.LR_SCHEDULERS,
                        help='Type of learning rate scheduler [default: no scheduler]')
    parser.add_argument("--lr_scheduler_args", dest='lr_scheduler_args', action=StoreDictKeyPair,
                        nargs="+", metavar="KEY=VAL", help="Arguments of the for the lr_scheduler. See PyTorch docs.")
    parser.add_argument('--w_recon_scheduler', default=None, type=str, choices=c.SCHEDULERS,
                        help='Type of scheduler for the reconstruction weight [default: no scheduler]')
    parser.add_argument("--w_recon_scheduler_args", dest='w_recon_scheduler_args', action=StoreDictKeyPair,
                        nargs="+", metavar="KEY=VAL", help="Arguments of the for the w_recon_scheduler.")

    # Other
    parser.add_argument('--seed', default=123, type=int, help='Seed value for torch, cuda, and numpy.')

    # Discriminative Model parameters (use only if --alg=Discriminative)
    parser.add_argument('--num_classes', default=5, type=int, help='Number of discrete labels for the classifier. Will be automatically set if you pass a labeling_fn')
    parser.add_argument('--labeling_fn', default=None, type=str, help='Function assigning binary labels to use.')
    parser.add_argument('--labelled_idx', default=None, type=int, nargs='+',  help='The indices annotated concepts to use (-1 means no annotated concepts, used in the paper)')
    parser.add_argument('--filter_fn', default=None, type=str, help='Postprocessing function that can remove parts of the dataset')

    # Correlation induction parameters to introduce correlation
    parser.add_argument('--corr_feature1', default=0, type=int, help='first feature index to correlate')
    parser.add_argument('--corr_feature2', default=1, type=int, help='second feature index to correlate')
    parser.add_argument('--sigma_corr', default=float("inf"), type=float, help='correlation strength parameter, inf means no correlation.')
    parser.add_argument('--n_cor', default=1, type=int, help='if filter_fn=normal_distr, gives how many pairs of variables should be correlated')
    parser.add_argument('--overwrite_cor_list', default=False, type=str2bool, help='if true and filter_fn=normal_distr, correlate corr_feature1 and corr_feature2 instead of the hardcoded list.')
    parser.add_argument('--oversampling_factor', default=3, type=int, help='how many batches to actually sample instead of one before for resampling the data in these batches acc. to the new weights.')
    parser.add_argument('--latent_reg', default = 0.0, type=float, help='Weight Regularization strength for the Model.')
    args = parser.parse_args(sys_args)

    print(args.encoder, args.decoder)
    if args.image_size != 64:
        print("Warning: certain decoders only work with image_size = 64")
    #assert args.image_size == 64, 'for now, models are hard coded to support only image size of 64x64'

    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)

    # test
    args = update_args(args) if args.test else args

    # make sure arguments for supplementary neural architectures are included
    if c.FACTORVAE in args.loss_terms:
        assert args.discriminator is not None, 'The FactorVAE algorithm needs a discriminator to test the ' \
                                               'permuted latent factors ' \
                                               '(try the flag: --discriminator=SimpleDiscriminator)'

    return args

def get_args_search(sysargs):
    """ Get arguements for the hyperparameter search. """
    parser = argparse.ArgumentParser(description='orthogonal attributions parameter search')
    # General Setup
    parser.add_argument('--recompute_baselines', default=False, type=bool, help='do not run optimization, only recompute the baselines.')

    # Hyperparameters
    parser.add_argument('--max_iter', default=int(1e6), type=int, help='maximum training iteration')
    parser.add_argument('--optimizer', default="adam", type=str, help='optimizer to use. adam, sgd, rmsprop')
    parser.add_argument('--encoder', default="SimpleGaussianConv64", type=str, help='Encoder Arch.')
    parser.add_argument('--decoder', default="SimpleConv64", type=str, help='Decoder Arch.')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate for optimizer')
    parser.add_argument('--batchsize', default = 24, type=int, help='batchsize')
    parser.add_argument('--num_directions', default = 6, type=int, help='Number of directions to discovers')
    parser.add_argument('--eval_batches', default = 10000, type=int, help='Number that should be sampled for computation of baselines.')
    parser.add_argument('--dset', default = "shapes3d", type=str, help='Dataset to use.')
    parser.add_argument('--evaluation_metric', default=None, type=str, choices=c.EVALUATION_METRICS, nargs='+', help='Metric to evaluate the model during training')
    parser.add_argument('--alg', default = "BetaVAE", type=str, help='model type, either BetaVAE or Discriminative')
    parser.add_argument('--skip_baselines', default=False, type=str2bool, help="If true, does not compute metrics for baseline methods")
    parser.add_argument('--image_size', default=64, type=int, help="size of input images, images are supposed to be squares.")
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')

    # Correlation induction
    parser.add_argument('--sigma_corr', default=float("inf"), type=float, help='correlation strength parameter, inf means no correlation.')
    parser.add_argument('--filter_fn', default=None, type=str, help='Function that filters out specific parts of the dataset')
    parser.add_argument('--corr_feature1', default=0, type=int, help='first feature to correlate')
    parser.add_argument('--corr_feature2', default=1, type=int, help='second feature to correlate')
    parser.add_argument('--n_cor', default=1, type=int, help='if filter_fn=normal_distr, gives how many pairs of variables should be correlated')
    parser.add_argument('--overwrite_cor_list', default=False, type=str2bool, help='if true and filter_fn=normal_distr, correlate corr_feature1 and corr_feature2 instead of the hardcoded list.')
    parser.add_argument('--oversampling_factor', default=3, type=int, help='how many batches to actually sample for resampling the data.')
    

    # Attribution methods
    parser.add_argument('--attribution', default = "ig", type=str, help='attribution method to use: grad, ig, sg, gen')
    parser.add_argument('--norm', default = True, type=str2bool, help='normalize attribution before computing loss')
    parser.add_argument('--disjoint', default = True, type=str2bool, help='use disjoint attributions (False = use orthogonal attributions)')
    parser.add_argument('--losstype', default = "offdiagl2", type=str, help='losstype to use for SGD methods: offdiagl2, detloss, logdetloss')

    # Paths
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('--evaliter', default = 2000, type=int, help='interval for evaluation')
    parser.add_argument('--logdir', default = "search_results", type=str, help='output save path')
    parser.add_argument('--dsetdir', default = ".", type=str, help='dataset directory from DLIB')
    parser.add_argument('--checkpointfileA', default = ".", type=str, help='path to the checkpoint A')
    parser.add_argument('--checkpointfileB', default = ".", type=str, help='path to the checkpoint B')

    args = parser.parse_args(sysargs)
    return args