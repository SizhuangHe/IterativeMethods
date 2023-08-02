import os
from random import randint, seed

import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import kneighbors_graph
from datasets import load_from_disk


class FMRIBrainActivityDataset(torch.utils.data.Dataset):
    """
    This dataloader loads brain activity readings from the UKBioBank dataset.
​
    fMRI recording signal windows make up node features, with 400 timepoints per node by default.
    Adjacency matrix is derived from a KNN graph on the xyz coordinates of each of the 424
    brain voxels in the parcellation.
​
    Task: signal reconstruction
    """
    def __init__(self,
                 ukb_arrow_dataset_path: str,
                 voxel_coordinates_arrow_dataset_path: str,
                 dataset_split="train",
                 recording_col_name="Subtract_Mean_Divide_Global_STD_Normalized_Recording",
                 timepoints_per_node: int = 100,
                 num_predicted_timepoints: int = 50,
                 knn_neighbors: int = 5
                 ):
        """
        Constructor for UKBioBank fMRI recording dataloader.
​
        Arguments:
            ukb_arrow_dataset_path:                 path to arrow dataset
            voxel_coordinates_arrow_dataset_path:   path to arrow dataset containing voxel coordinates
            dataset_split:                          train, val, or test
            recording_col_name:                     column name containing normalized recording
            timepoints_per_node:                    number of timepoints per node
            num_predicted_timepoints:               number of timepoints to predict into future
            knn_neighbors:                          number of nearest neighbors in KNN adjacency graph.
        """
        assert dataset_split in ["train", "val", "test"]
        self.dataset_split = dataset_split
        self.recording_col_name = recording_col_name
        self.timepoints_per_node = timepoints_per_node
        self.num_predicted_timepoints = num_predicted_timepoints
        self.knn_neighbors = knn_neighbors

        # Load Dataset
        recording_ds = load_from_disk(ukb_arrow_dataset_path)
        coords_ds = load_from_disk(voxel_coordinates_arrow_dataset_path)
        self.recording_ds = recording_ds
        self.coords_ds = coords_ds
        self.voxel_names = ["Voxel {}".format(idx) for idx in range(coords_ds.num_rows)]

        # Create KNN Adjacency Graph Using Brain Region Coordinates
        x_list = [coords_ds[voxel_idx]["X"] for voxel_idx in range(coords_ds.num_rows)]
        y_list = [coords_ds[voxel_idx]["Y"] for voxel_idx in range(coords_ds.num_rows)]
        z_list = [coords_ds[voxel_idx]["Z"] for voxel_idx in range(coords_ds.num_rows)]
        xyz_coords_np = np.array([x_list, y_list, z_list], dtype=np.float32).T
        adjacency = kneighbors_graph(X=xyz_coords_np, n_neighbors=knn_neighbors, include_self=True)
        adj_arr = adjacency.toarray().astype(np.int64)
        adjacency_t = torch.tensor(adj_arr, dtype=torch.int64)
        self.edge_idx, _ = dense_to_sparse(adjacency_t)

        print("Constructor finished.\n\n")

    def __len__(self):
        return self.recording_ds.num_rows

    def __getitem__(self, idx):
        """
        Function to retrieve a sample from one fMRI recording. Function will load a dataset sample, pick a
        start signal index, and then take a window of N points, with ground truth points being the M points
        following N since the task is to predict timepoints into the future.
​
        Args:
            idx: index of sample to load from arrow dataset
​
        Returns:
            signal_window:          numpy array of shape [num_voxels, timepoints_per_node]
            ground_truth_labels:    numpy array of shape [num_voxels, num_predicted_timepoints]
            window_timepoints:      numpy array of shape [num_voxels, timepoints_per_node]
        """
        recording = self.recording_ds[idx][self.recording_col_name]
        recording = np.array(recording, dtype=np.float32)
        recording = recording.T  # shape [424 voxels, recording_len=490 or 500+]
        num_voxels, recording_len = recording.shape

        # Pick random window of length timepoints_per_node in recording
        if self.dataset_split == "train":
            start_idx = randint(0, recording_len - self.timepoints_per_node - self.num_predicted_timepoints)
        else:
            start_idx = 0  # If test dataset, for reproducibility, always give first N timepoints
        end_idx = start_idx + self.timepoints_per_node
        signal_window = recording[:, start_idx: end_idx]  # shape [num_voxels, timepoints_per_node]

        # Get ground truth labels (next M timepoints)
        ground_truth_labels = recording[:, end_idx: end_idx + self.num_predicted_timepoints]

        # Get corresponding time indices in recording
        window_timepoints = np.expand_dims(np.arange(start_idx, end_idx), 0)
        window_timepoints = np.repeat(window_timepoints, num_voxels, axis=0)  # same size as signal window

        return signal_window, ground_truth_labels, window_timepoints


if __name__ == "__main__":
    if os.getcwd().split("/")[-1] == "neurolib_sparse_irregular":
        os.chdir("../..")

    random_seed = 1234
    torch.manual_seed(random_seed)
    seed(random_seed)
    np.random.seed(random_seed)

    #--- Paths ---#
    # Train dataset: /home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/train_ukbiobank1000
    # Validation dataset: /home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/val_ukbiobank1000
    # Test dataset: /home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/test_ukbiobank1000
    # Brain region coordinates: /home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/Brain_Region_Coordinates
    dataset = FMRIBrainActivityDataset(
        ukb_arrow_dataset_path="home/vast/palmer/scratch/dijk/sr2464/datasets/UKBioBank1000_Arrow_v4/train_ukbiobank1000",
        voxel_coordinates_arrow_dataset_path="/home/sr2464/palmer_scratch/datasets/UKBioBank1000_Arrow_v4/Brain_Region_Coordinates",
        dataset_split="train",
        recording_col_name="Subtract_Mean_Divide_Global_STD_Normalized_Recording",
        timepoints_per_node=100,
        num_predicted_timepoints=50,
        knn_neighbors=5
    )

    # Get a sample of dataset to check
    sample_signal_window, sample_ground_truth_labels, sample_window_timepoints = next(iter(dataset))
    print("Done.")