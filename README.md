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
To train the models that we used in our work, we provide scripts in the folder ``scripts``. For training the models with two correlated factors in Section 4.2, we provide four scripts named ```3dshapes_hypersearch<ModelArchitecture>_fact.sh```.
For example, they can be called as follows:
```
export OUTPUT="."; export DSET_PATH="datasets/Disentanglement"; ./scripts/3dshapes_hypersearchBetaVAE_fact.sh 4 0.7 1 0 0
```

The first part of the command sets the corresponding environment variables to store the output and to read the datasets respectively.

The five arguments to the script are 
* The hyperparameter value. See Table 4 in the appendix of our paper for details.
* The correlation strength. We use the values 0.2, 0.4 (in the main paper), 0.7 and inf.
* The index of the first correlated factor
* The index of the second correlated factor. We use the combintations (1, 0 = floor, background / 5, 0 = orientation, background / 3, 5 = size, orientation) on 3dshapes.
* A suffix that is appended to the output file to indicate the run number during multiple executions of the script (could be 0, 1, 2, ... or A, B, C, ...)

### Running the post-hoc concept discovery IMA/DMA

## Experiments with Discriminative Models (Section 4.4)

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
