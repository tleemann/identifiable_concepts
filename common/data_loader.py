import os
import numpy as np
import logging

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from data.shapes3d_gaussian import Shapes3DGaussian
from data.four_bars import FourBars, ColorBar
from common import constants as c

class LabelHandler(object):
    def __init__(self, labels, label_weights, class_values):
        self.labels = labels
        self._label_weights = None
        self._num_classes_torch = torch.tensor((0,))
        self._num_classes_list = [0]
        self._class_values = None
        if labels is not None:
            self._label_weights = [torch.tensor(w) for w in label_weights]
            self._num_classes_torch = torch.tensor([len(cv) for cv in class_values])
            self._num_classes_list = [len(cv) for cv in class_values]
            self._class_values = class_values

    def label_weights(self, i):
        return self._label_weights[i]

    def num_classes(self, as_tensor=True):
        if as_tensor:
            return self._num_classes_torch
        else:
            return self._num_classes_list

    def class_values(self):
        return self._class_values

    def get_label(self, idx):
        if self.labels is not None:
            return torch.tensor(self.labels[idx], dtype=torch.long)
        return None

    def has_labels(self):
        return self.labels is not None


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform, labels, label_weights, name, class_values, num_channels, seed):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))
        self._num_channels = num_channels
        self._name = name
        self.seed = seed

        self.label_handler = LabelHandler(labels, label_weights, class_values)

    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform(img1)

        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
        return img1, label1


class CustomNpzDataset(Dataset):
    def __init__(self, data_images, transform, labels, label_weights, name, class_values, num_channels, seed):
        self.seed = seed
        self.data_npz = data_images
        self._name = name
        self._num_channels = num_channels

        self.label_handler = LabelHandler(labels, label_weights, class_values)

        self.transform = transform
        self.indices = range(len(self))

    @property
    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        img1 = Image.fromarray(self.data_npz[index1] * 255)
        if self.transform is not None:
            img1 = self.transform(img1)

        label1 = 0
        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
        return img1, label1

    def __len__(self):
        return self.data_npz.shape[0]


class DisentanglementLibDataset(Dataset):
    """
    Data-loading from Disentanglement Library

    Note:
        Unlike a traditional Pytorch dataset, indexing with _any_ index fetches a random batch.
        What this means is dataset[0] != dataset[0]. Also, you'll need to specify the size
        of the dataset, which defines the length of one training epoch.

        This is done to ensure compatibility with disentanglement_lib.
    """

    def __init__(self, name, seed=0, labeling_fn=None):
        """
        Parameters
        ----------
        name : str
            Name of the dataset use. You may use `get_dataset_name`.
        seed : int
            Random seed.
        labeling_fn: Function to assign lables or None.
        """
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        
        self.dataset = _get_base_dlib_dataset_by_name(name)
        
        self.iterator_len = self.dataset.images.shape[0]
        if labeling_fn is not None:
            self.label_gen_function = labeling_fn
        else:
            self.label_gen_function = lambda x: 0 # Label generating function

    @staticmethod
    def has_labels():
        return False

    def num_channels(self):
        return self.dataset.observation_shape[2]

    def __len__(self):
        return self.iterator_len

    def sample_observations_from_factors(self, factors):
        return self.dataset.sample_observations_from_factors(factors, random_state=self.random_state)
        
    def __getitem__(self, item):
        assert item < self.iterator_len
        factors, output = self.dataset.sample(1, random_state=self.random_state)
        # Remove auxiliary dimension.
        factors = factors.flatten()
        if len(output.shape) == 4:
            output = output.reshape(output.shape[1:])
        # Convert output to CHW from HWC
        return torch.from_numpy(factors), torch.from_numpy(np.moveaxis(output, 2, 0), ).type(torch.FloatTensor), self.label_gen_function(torch.from_numpy(factors))

class ResamplingDataLoader():
    """ Implement a resampling strategy that induces correlation between ground truth factors in the data loader.
    We previously assume a uniform distribution (or pass other input_dens) p(x) of the ground truth factors.
    However, we would like to introduce a distribution q(x) (output_dens) Therefore, we adapt the probability of each point being sampled
    by q(x)/p(x). Distributions need not to be normalized. 
    The resampling is achieved by loading a larger batch and computing the new probabilities for each sample in the batch.
    The data that is actually returned is sampled from the new probability distribution with replacement.

    The underlying dataset must return a tuple of the form (gt_factors, inputs, ...)
    """
    def __init__(self, base_dl, oversampling_factor = 3, output_dens = lambda x: torch.ones(len(x)), input_dens = lambda x: torch.ones(len(x))):
        """ 
            Parameters:
                base_dl: torch.DataLoader of the underlying distribution
                oversampling_factor: This argument actual batches are drawn to recompute the probabilities and return one batch of outputs.
                output dens: lambda representing q
                input dens: labmbda representing p
        """
        self.my_dl = base_dl
        self.oversampling_factor = oversampling_factor
        self.input_p = input_dens
        self.output_q = output_dens

    def __iter__(self):
        class ResampleIterator():
            def __init__(self, dataload_obj):
                """dataload_obj: The ResamplingDataloader. """
                self.num_called = 0
                self.base_dl = dataload_obj
                self.base_iter = dataload_obj.my_dl.__iter__()

            def __next__(self):
                self.num_called += 1
                if self.num_called > len(self.base_dl):
                    raise StopIteration
            
                batch_list = [] # list of tuples.
                # Sample oversampling_factor number of batches
                for i in range(self.base_dl.oversampling_factor):
                    batch_list.append(self._get_next_batch())

                cat_batch_list = []
                for k in range(len(batch_list[i])):
                    tup = torch.cat([r[k] for r in batch_list], dim=0)
                    cat_batch_list.append(tup)
                #print([s.shape for s in cat_batch_list])
                # Compute density ratios.
                px = self.base_dl.input_p(cat_batch_list[0])
                qx = self.base_dl.output_q(cat_batch_list[0])
                ratio = qx/(px+1e-6)
                sample_ind = torch.multinomial(ratio, self.base_dl.batch_size, replacement=True)
                return tuple([r[sample_ind] for r in cat_batch_list])
                
            def _get_next_batch(self):
                try:
                    return self.base_iter.__next__()
                except StopIteration:
                    self.base_iter = self.base_dl.my_dl.__iter__()
                    return self.base_iter.__next__()
        return ResampleIterator(self)

    def __len__(self):
        return len(self.my_dl)

    @property
    def dataset(self):
        return self.my_dl.dataset 
    
    @property
    def batch_size(self):
        return self.my_dl.batch_size 


def _get_dataloader_with_labels(name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                                shuffle, droplast):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), ])
    labels = None
    label_weights = None
    label_idx = None
    label_names = None
    class_values = None

    # check if labels are provided as indices or names
    if include_labels is not None:
        try:
            int(include_labels[0])
            label_idx = [int(s) for s in include_labels]
        except ValueError:
            label_names = include_labels
    logging.info('include_labels: {}'.format(include_labels))

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        labels_file = os.path.join(root, 'list_attr_celeba.csv')

        # celebA images are properly numbered, so the order should remain intact in loading
        labels = None
        if label_names is not None:
            labels = []
            labels_all = np.genfromtxt(labels_file, delimiter=',', names=True)
            for label_name in label_names:
                labels.append(labels_all[label_name])
            labels = np.array(labels).transpose()
        elif label_idx is not None:
            labels_all = np.genfromtxt(labels_file, delimiter=',', skip_header=True)
            labels = labels_all[:, label_idx]

        if labels is not None:
            # celebA labels are all binary with values -1 and +1
            labels[labels == -1] = 0
            from pathlib import Path
            num_l = labels.shape[0]
            num_i = len(list(Path(root).glob('**/*.jpg')))
            assert num_i == num_l, 'num_images ({}) != num_labels ({})'.format(num_i, num_l)

            # calculate weight adversely proportional to each class's population
            num_labels = labels.shape[1]
            label_weights = []
            for i in range(num_labels):
                ones = labels[:, i].sum()
                prob_one = ones / labels.shape[0]
                label_weights.append([prob_one, 1 - prob_one])
            label_weights = np.array(label_weights)

            # all labels in celebA are binary
            class_values = [[0, 1]] * num_labels

        data_kwargs = {'root': root,
                       'labels': labels,
                       'label_weights': label_weights,
                       'class_values': class_values,
                       'num_channels': 3}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites_full':
        root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        npz = np.load(root)

        if label_idx is not None:
            labels = npz['latents_values'][:, label_idx]
            if 1 in label_idx:
                index_shape = label_idx.index(1)
                labels[:, index_shape] -= 1

            # dsprite has uniformly distributed labels
            num_labels = labels.shape[1]
            label_weights = []
            class_values = []
            for i in range(num_labels):
                unique_values, count = np.unique(labels[:, i], axis=0, return_counts=True)
                weight = 1 - count / labels.shape[0]
                if len(weight) == 1:
                    weight = np.array(1)
                else:
                    weight /= sum(weight)
                label_weights.append(np.array(weight))

                # always set label values to integers starting from zero
                unique_values_mock = np.arange(len(unique_values))
                class_values.append(unique_values_mock)
            label_weights = np.array(label_weights)

        data_kwargs = {'data_images': npz['imgs'],
                       'labels': labels,
                       'label_weights': label_weights,
                       'class_values': class_values,
                       'num_channels': 1}
        dset = CustomNpzDataset
    else:
        raise NotImplementedError
    data_kwargs.update({'seed': seed,
                        'name': name,
                        'transform': transform})
    dataset = dset(**data_kwargs)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=droplast)

    if include_labels is not None:
        logging.info('num_classes: {}'.format(dataset.num_classes(False)))
        logging.info('class_values: {}'.format(class_values))

    return data_loader

def _get_base_dlib_dataset_by_name(name: str, eval=False):
    """ Return a DisentanglementLib Dataset object by name. """
    from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
    if name == "shapes3d_toy":
        return Shapes3D()
    elif name == "shapes3d_noisy":
        return Shapes3DGaussian(name="noisy")
    elif name == "shapes3d_gaussian":
        return Shapes3DGaussian()
    elif name == "fourbars":
        return FourBarsSimple(11)
    elif name == "colorbar":
        return BarToy(11)
    else:
        return get_named_ground_truth_data(name)

def get_dataset_name(name):
    """Returns the name of the dataset from its input argument (name) or the
    environment variable `AICROWD_DATASET_NAME`, in that order."""
    return name or os.getenv('AICROWD_DATASET_NAME', c.DEFAULT_DATASET)


def get_datasets_dir(dset_dir):
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    return dset_dir or os.getenv('DISENTANGLEMENT_LIB_DATA')


def _get_dataloader(name, batch_size, seed, num_workers, pin_memory, shuffle, droplast, labeling_fn):
    """
    Makes a dataset using the disentanglement_lib.data.ground_truth functions, and returns a PyTorch dataloader.
    Image sizes are fixed to 64x64 in the disentanglement_lib.
    :param name: Name of the dataset use. Should match those of disentanglement_lib
    :return: DataLoader
    """
    dataset = DisentanglementLibDataset(name, seed=seed, labeling_fn = labeling_fn)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast, pin_memory=pin_memory,
                        num_workers=num_workers)
    
    return loader


def get_dataloader(dset_name, dset_dir, batch_size, seed, num_workers, image_size, include_labels, pin_memory,
                   shuffle, droplast, resample_fn = None, oversampling_factor = 3, labeling_fn = None):
    """ Return a data loader.
        One batch consists of 2 to 3 items, factors, input and (if available) label.
        labeling_fn: Function to assign labels dependent on ground truth factors.
    """
    dset_name = get_dataset_name(dset_name)
    dsets_dir = get_datasets_dir(dset_dir)

    logging.info(f'Datasets root: {dset_dir}')
    logging.info(f'Dataset: {dset_name}')

    base_dl = None
    # use the dataloader of Google's disentanglement_lib
    base_dl = _get_dataloader(dset_name, batch_size, seed, num_workers, pin_memory, shuffle, droplast, labeling_fn)

    if resample_fn is not None:
        return ResamplingDataLoader(base_dl, oversampling_factor, resample_fn)
    else:
        return base_dl



    