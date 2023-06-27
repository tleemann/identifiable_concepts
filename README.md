# When Are Post-hoc Conceptual Explanations Identifiable?

This repository contains the code for the Paper ["When Are Post-Hoc Conceptual Explanations Identifiable?"](https://arxiv.org/abs/2206.13872) (accepted at UAI 2023)

**Note: Work still in progress. Code will be made available soon.**

## Setup
The following steps are required to run the code in this repository using a dedicated anaconda environment.

### Creating an Anaconda Environment
Make sure you have a working installation of anaconda on your system and go to the main directory of this repository in your terminal.
Then install the requirements into a new conda environment named ```identifiable_concepts``` run the following commands 
```conda env create -f environment.yml
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
