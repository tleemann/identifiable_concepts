# When Are Post-hoc Conceptual Explanations Identifiable?

This repository contains the code for the Paper ["When Are Post-Hoc Conceptual Explanations Identifiable?"](https://arxiv.org/abs/2206.13872) (accepted at UAI 2023)

**Note: Work still in progress. Code will be made available soon.**

## Setup
The following steps are required to run the code in this repository using a dedicated anaconda environment.

### Creating an Anaconda Environment
Make sure you have a working installation of anaconda on your system and go to the main directory of this repository in your terminal.
Then install the requirements into a new conda environment named ```identifiable_concepts``` run the following commands 
```
conda env create -f environment.yml
```

Then run
```
conda activate identifiable_concepts
```

### Downloading datasets
The 3d shapes and the MPI3d datasets need to be downloaded before they can be used with this repository.

**3DShapes**
3DShapes has to be downloaded and preprocessed additionally, as it cannot be automatically loaded by disentanglement_lib. Go to the folder where you downloaded the other data sets and ``mkdir 3dshapes``. Download the data file ``3dshapes.h5`` from [here](https://console.cloud.google.com/storage/browser/3d-shapes) in this folder.
Then run the script ``scripts/3dshapes_to_npz.py`` in this repo from the folder ``3dshapes``, which will convert the ``3dshapes.h5`` into the correct format for disentanglement_lib to read it.

**MPI3D**
[Todo: add instructions for MPI3d]

## Experiments with Toy datasets (Section 4.1)

## Experiments with Disentanglement Models (Section 4.2, Section 4.3)

### Training models
To train the models that we used in our work in Section 4.2 and 4.3, we provide scripts in the folder ``scripts``. First, we train the models with correlated factors in four scripts named ```3dshapes_hypersearch<ModelArchitecture>_fact.sh```.
For example, they can be called as follows:
```
export OUTPUT="."; export DSET_PATH="datasets/Disentanglement"; ./scripts/3dshapes_hypersearchBetaVAE_fact.sh 4 0 pair 0.7 1 0 0
```

The first part of the command sets the corresponding environment variables to store the output and to read the datasets respectively.

The seven arguments to the script are 
* The hyperparameter value of the method. See Table 4 in the appendix of our paper for details.
* A suffix that is appended to the output file to indicate the run number during multiple executions of the script (could be 0, 1, 2, ... or A, B, C, ...)
* Which correlation to use: `normal_distr` for a normal distribution and `pair` for correlating just two factors
* The correlation strength. We use the values 0.2, 0.4 (in the main paper), 0.7 and inf. Note that for `pair`, lower means more correlated and inf means uncorrelated, whereas for `normal_distr`, 0 means uncorrelated and 1 means fully correlated.
* In case of `pair`: The index of the first correlated factor
* In case of `pair`: The index of the second correlated factor. We use the combintations (1, 0 = floor, background / 5, 0 = orientation, background / 3, 5 = size, orientation) on 3dshapes.
* In case of `normal_distr`: how many pairwise correlations to add to the covariance matrix (0 to 15). Note that for some combinations, the correlation strength is lowered automatically until the covariance matrix is positive definite. The script will log this.

### Running the post-hoc concept discovery IMA/DMA

To run IMA and DMA, we provide the script ```discover_concepts.py```, which implements our methods.
To discover concept post-hoc on trained disentanglement models with the scripts in the folder ```scripts```discussed in the previous section, we can use the script ```3dshapes_PosthocDiscover.sh``` as follows (again setting some environment variables) by running

```
export LOGDIR="search_results"; \
export DSET_PATH="datasets/Disentanglement"; \
export CHECKPOINTDIR="checkpoints"; \
./scripts/3dshapes_PosthocDiscover.sh BetaTCVAE_wtc-10_Corr-0.4_R0 grad pair 0.4 0 1 0
```

in the terminal. 

The seven arguments to the script are 

* The name of the checkpoint created by the training script
* The attribution method used to compute the gradient matrix $J_f$ (default ```grad``` used and described in the paper)
* Which correlation to use: `normal_distr` for a normal distribution and `pair` for correlating just two factors
* The correlation strength. We use the values 0.2, 0.4 (in the main paper), 0.7 and inf. Note that for `pair`, lower means more correlated and inf means uncorrelated, whereas for `normal_distr`, 0 means uncorrelated and 1 means fully correlated.
* In case of `pair`: The index of the first correlated factor
* In case of `pair`: The index of the second correlated factor. We use the combintations (1, 0 = floor, background / 5, 0 = orientation, background / 3, 5 = size, orientation) on 3dshapes.
* In case of `normal_distr`: how many pairwise correlations to add to the covariance matrix (0 to 15). Note that for some combinations, the correlation strength is lowered automatically until the covariance matrix is positive definite. The script will log this.

## Experiments with Discriminative Models (Section 4.4)

To train a discriminative models used in our work, we provide the script ```3dshapes_Discriminative.sh``` in the folder ```scripts```. It can be called as follows

```
export OUTPUT="."; \
export DSET_PATH="datasets/Disentanglement"; \
./scripts/3dshapes_Discriminative.sh 0.2 -1 0 
```

where the three arguments provided to the script are
* The correlation strength. We use the values 0.1, 0.15, 0.2, and inf.
* Our code provides the opportunity to pass partial factor annotations to the model. However, we do not use this feature for this work and pass a value of -1 (meaning that the training works without any factors annotations in a fully unsupervised manner.)
*  A suffix that is appended to the output file to indicate the run number during multiple executions of the script (could be 0, 1, 2, ... or A, B, C, ...)

### Post-hoc concept discovery on the discriminative models.

Having trained a discriminative model with the above instructions, we can run our post-hoc methods with the following script

```
export LOGDIR="search_results"; \
export DSET_PATH="datasets/Disentanglement"; \
export CHECKPOINTDIR="checkpoints"; \
./scripts/3dshapes_DiscriminativePosthoc.sh 0.2 -1 0 grad
```

where the four arguments provided to the script are
* The correlation strength. We use the values 0.1, 0.15, 0.2, and inf.
* Our code provides the opportunity to pass partial factor annotations to the model. However, we do not use this feature for this work and pass a value of -1 (meaning that the training works without any factors annotations in a fully unsupervised manner.)
*  A suffix that is appended to the output file to indicate the run number during multiple executions of the script (could be 0, 1, 2, ... or A, B, C, ...)
* The attribution method used to compute the gradient matrix $J_f$ (default ```grad``` used in the paper)

The resulting evaluation scores will be stored in the folder ```search_results```.

## Experiments on CUB-200 (Section 4.5)


## Credits
We would like to thank the authors of several other code bases that contributed to this work.
This repository was built on the [```disentanglement-pytorch```]((https://github.com/amir-abdi/disentanglement-pytorch) repository.

Most of the evaluation metrics are taken from [```disentanglement_lib```](https://github.com/google-research/disentanglement_lib).

## Citation
Please cite us if you find our work insightful or use our ressources in your own work, for instance with the following BibTex entry:

```
@article{leemann2022when,
  title={When are Post-hoc Conceptual Explanations Identifiable?},
  author={Leemann, Tobias and Kirchhof, Michael and Rong, Yao and Kasneci, Enkelejda and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2206.13872},
  year={2022}
}
```
