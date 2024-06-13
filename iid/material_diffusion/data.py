import os

import torch
from batch import Batch

from iid.data import IIDDataset


class InteriorVerseDataset(IIDDataset):
    FEATURES = ["im", "albedo", "normal", "depth", "material", "mask"]
    DERIVED_FEATURES = []

    DEPTH_SCALING = 1000

    def load_dataset(self, allow_missing_features=False):
        # Collect the data
        data = Batch()

        # Collect the scene list
        scene_list = []
        for scene_folder_path in self.split_list:
            # Check if the scene folder exists
            if os.path.exists(os.path.join(self.root, scene_folder_path)):
                scene_list.append(scene_folder_path)

        # Collect the features
        self.module_logger.debug("Collecting features")
        data['samples'] = Batch(default=Batch)
        data['sample_ids'] = []

        for scene_folder in scene_list:
            scene_folder_path = os.path.join(self.root, scene_folder)
            for file_name in sorted(os.listdir(scene_folder_path)):
                if "_" not in file_name:
                    continue

                view_id = file_name.split('_')[0]
                sample_id = os.path.join(scene_folder, view_id)

                if sample_id not in data['samples']:
                    data['sample_ids'].append(sample_id)
                    for feature in self.features_to_include:
                        data['samples'][sample_id][feature] = os.path.join(scene_folder, f"{view_id}_{feature}.exr")

        # Sanity check
        lengths = [len(list(data['samples'][sample_id].keys())) for sample_id in data['samples'].keys()]
        assert all([lengths[0] == l for l in lengths]), "Missing feature!"

        return data


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Adapted from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, indices):
        """
        Creates new sampler
        :param indices: The indices to sample from sequentially
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
