import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LipDataset(Dataset):
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    sensor_size = (224, 90, 2)

    def __init__(
        self,
        path,
        used_samples_percentage=1.0,
        augmentation=False,
        transform=None,
        target_transform=None,
        exclude_files=[],
        random_seed=42,
    ):
        super(LipDataset, self).__init__()

        random.seed(random_seed)

        self.transform = transform
        self.target_transform = target_transform

        self.class_names = []

        self.samples_per_class = {}

        self.files = []
        self.file_labels = []

        self.data_dir = path + "/train10/"

        for root, dirs, _ in os.walk(self.data_dir):
            for dir in dirs:
                class_name = dir

                if class_name not in self.class_names:
                    self.samples_per_class[class_name] = 0
                    self.class_names.append(class_name)

                class_label = self.class_names.index(class_name)

                csv_files = [
                    file
                    for file in os.listdir(os.path.join(root, dir))
                    if file.endswith(".csv")
                ]

                picked_files = random.sample(
                    csv_files,
                    int(used_samples_percentage * len(csv_files)),
                )

                for file in picked_files:
                    path = os.path.join(root, dir, file)

                    if path in exclude_files:
                        continue

                    self.files.append(path)
                    self.file_labels.append(class_label)
                    self.samples_per_class[class_name] += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        target = self.file_labels[idx]

        events_input = read_lip_events(self.files[idx])

        if self.transform is not None:
            events_input = self.transform(events_input)
        if self.target_transform is not None:
            events_input = self.target_transform(target)

        return events_input, target

    def print_infos(self):
        print(f"Total samples count : {self.__len__()}")
        print("Samples count per class :")
        for c in self.class_names:
            print(f"    Class {c}: {self.samples_per_class[c]} samples")


def read_lip_events(path):
    events = np.loadtxt(path, delimiter=",", skiprows=1)
    # swap p and t
    events[:, [2, 3]] = events[:, [3, 2]]

    # remove events where abs(t) >= mean + 2 * std
    t = events[:, 2]
    mean = np.mean(t)
    std = np.std(t)
    events = events[abs(t - mean) < 2 * std]

    events_input = np.fromiter(map(tuple, events), dtype=LipDataset.dtype)
    return events_input
