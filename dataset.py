import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable, List


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:
            aux_path = f"{dataset_path}/{subdir}"

            # Read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(rgb_paths, ground_truth_data)

            for i in range(sequence_length, len(rgb_paths) + 1):
                sequence_images = rgb_paths[i - sequence_length:i]
                
                if not validation:
                    ground_truth = [gt[1] for gt in interpolated_ground_truth[i - sequence_length:i]]
                else:
                    ground_truth = []

                self.sequences.append({
                    "images": sequence_images,
                    "ground_truth": ground_truth,
                    "timestamp": rgb_paths[i - 1][0]
                })

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:

        # Load sequence of images
        sequence_images = []
        ground_truth_pos = []
        timestamp = self.sequences[idx]["timestamp"]

        for _, image_path in self.sequences[idx]["images"]:
            rgb_img = cv2.imread(image_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                rgb_img = self.transform(rgb_img)
                
            sequence_images.append(rgb_img)
        
        sequence_images = torch.stack(sequence_images)
        
        if not self.validation:
            ground_truth_pos = torch.FloatTensor([
                self.sequences[idx]["ground_truth"][-1][j] - self.sequences[idx]["ground_truth"][-2][j]
                for j in range(7)
            ])
        else:
            ground_truth_pos = torch.FloatTensor([])

        return sequence_images, ground_truth_pos, timestamp

    def read_images_paths(self, dataset_path: str) -> List[Tuple[float, str]]:
        paths = []
        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:
                if line.startswith("#"):  # Skip comment lines
                    continue
                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"
                paths.append((timestamp, image_path))
        return paths

    def read_ground_truth(self, dataset_path: str) -> List[Tuple[float, List[float]]]:
        ground_truth_data = []
        with open(f"{dataset_path}/groundtruth.txt", "r") as file:
            for line in file:
                if line.startswith("#"):  # Skip comment lines
                    continue
                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))
        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: List[Tuple[float, str]],
            ground_truth_data: List[Tuple[float, List[float]]]
    ) -> List[Tuple[float, List[float]]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:
            nearest_idx = np.argmin(np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))
            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
