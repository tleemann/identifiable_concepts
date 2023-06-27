# Strings
LOSS = 'loss'
ACCURACY = 'acc'
ITERATION = 'iteration'
WANDB_NAME = 'disentanglement'
INPUT_IMAGE = 'input_image'
RECON_IMAGE = 'recon_image'
RECON = 'recon'
FIXED = 'fixed'
SQUARE = 'square'
ELLIPSE = 'ellipse'
HEART = 'heart'
TRAVERSE = 'traverse'
RANDOM = 'random'
TEMP = 'tmp'
GIF = 'gif'
JPG = 'png'
BMP = 'bmp'
FACTORVAE = 'FactorVAE'
DIPVAEI = 'DIPVAEI'
DIPVAEII = 'DIPVAEII'
BetaTCVAE = 'BetaTCVAE'
INFOVAE = 'InfoVAE'
TOTAL_VAE = 'total_vae'
CLASSIFICATION_LOSS = 'classification_loss'
CONCEPT_LOSS = 'concept_loss'
REG_LOSS = 'reg_loss'
TOTAL_VAE = 'total_vae'
TOTAL_VAE_EPOCH = 'total_vae_epoch'
LEARNING_RATE = 'learning_rate'

# Supported Arguements for this version.
# Algorithms
ALGS = ('AE', 'VAE', 'BetaVAE', 'Discriminative')
LOSS_TERMS = (FACTORVAE, DIPVAEI, DIPVAEII, BetaTCVAE, INFOVAE)

# Datasets
DATASETS = ('shapes3d', 'shapes3d_noisy', 'shapes3d_gaussian','mpi3d_real', 'fourbars', 'colorbar')
DEFAULT_DATASET = DATASETS[0]  # shapes3d
TEST_DATASETS = DATASETS[0:2]  # celebA, dsprites_full

# Architectures
DISCRIMINATORS = ('SimpleDiscriminator', 'SimpleDiscriminatorConv64')
CLASSIFIERS = ('SimpleClassifier','GAPClassifier')
TILERS = ('MultiTo2DChannel',)
DECODERS = ('SimpleConv64', 'ShallowLinear', 'DeepLinear')
ENCODERS = ('SimpleConv64', 'SimpleGaussianConv64', 'PadlessConv64', 'PadlessGaussianConv64',
            'ShallowGaussianLinear', 'DeepGaussianLinear', 'SimpleCAMGaussianConv64')


# Evaluation Metrics
EVALUATION_METRICS = ('dci', 'factor_vae_metric', 'sap_score', 'mig', 'accuracy', 'dirscore', 'dirscore_bin')

# Schedulers
LR_SCHEDULERS = ('ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                 'CosineAnnealingLR', 'CyclicLR', 'LambdaLR')
SCHEDULERS = ('LinearScheduler', )
