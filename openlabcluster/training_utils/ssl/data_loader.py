"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import os
import shutil
import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class FeatureNameKeyError(KeyError):
    pass


class DataFileLoadingError(FileNotFoundError):
    pass


def get_data_paths(root_path: str, data_path: str, train_files: str):
    """Organizes data directories

    Inputs:
        root_path: the root directory of dataset
        data_path: the name of dataset folder
        train_files: a list of training dataset filenames or a string
    Outputs:
        a list of absolute directory of training files
    """
    if type(train_files) == str:
        return [os.path.join(root_path, data_path, train_files)]

    return [os.path.join(root_path, data_path, file) for file in train_files]


def get_data_list(data_paths: list, indices=None, keypoint_names: list = None, return_video=False, videos_exist=False,
                  max_data_len=None):
    """ Extracts keypoint sequences, labels, and corresponding video names from several data directories

    Inputs:
        data_paths: a list, contains data file directories (h5, h5py files).
        indices: a list, contains indices of files to extract in the data_paths (extracts all if None)
        keypoint_names: the name of keypoints
        return_video: if true, returns the video names
        videos_exits: if true, video exists for each file in the data_path
    Returns:
        data_list: a list of keypoint sequences, each sequence is in shape (sequence length, feature length)
        label_list: a list containing labels for each keypoint sequence (a placeholder by default 0 if none of samples is labeled)
    """

    data_list = []
    label_list = []
    video_list = []

    for data_path in data_paths:
        try:
            f = h5py.File(data_path, 'r')
        except:
            raise DataFileLoadingError(data_path)

        if 'df_with_missing' in list(f.keys()):
            # data input assumed in DeepLabCut output format
            video_list.append(data_path)
            data = np.array([])

            df = pd.read_hdf(data_path)
            scorers = set([key[0] for key in df.keys()])

            for scorer in scorers:
                # Slices out the data in the dataframe
                idx = pd.IndexSlice

                if keypoint_names is None:
                    datapoint = df[scorer].loc[idx[:], idx[:, ('x', 'y')]].to_numpy()
                else:
                    try:
                        datapoint = df[scorer][keypoint_names].loc[idx[:], idx[:, ('x', 'y')]].to_numpy()
                    except KeyError as e:
                        raise FeatureNameKeyError(str(e))

                if len(data) == 0:
                    data = datapoint
                else:
                    np.concatenate((data, datapoint), axis=0)

            num_keypoints = data.shape[1]

            # Preprocesses data.
            y_cor = np.arange(0, num_keypoints, 2)
            x_cor = np.arange(1, num_keypoints, 2)
            # Normalize along dimension.
            max_26 = np.amax(data, axis=0)
            min_26 = np.amin(data, axis=0)
            max_x = np.max([max_26[i] for i in range(1, num_keypoints, 2)])
            max_y = np.max([max_26[i] for i in range(0, num_keypoints, 2)])
            min_x = np.min([min_26[i] for i in range(1, num_keypoints, 2)])
            min_y = np.min([min_26[i] for i in range(0, num_keypoints, 2)])

            data[:, y_cor] = (data[:, y_cor] - min_y) / (max_y - min_y)
            data[:, x_cor] = (data[:, x_cor] - min_x) / (max_x - min_x)

            x = torch.tensor(data, dtype=torch.float)
            y = 0

            data_list.append(x)
            label_list.append(y)

        else:
            output = load_preprocessed_h5_datafile(data_path, indices=indices, videos_exist=videos_exist)

            data_list += output[0]  # data list
            label_list += output[1]  # label list

            if videos_exist:
                video_list += output[2]
            else:
                video_list += [data_path + '-clip' + str(i) for i in range(len(output[0]))]

    if max_data_len is not None:
        data_list = data_list[:max_data_len]
        label_list = label_list[:max_data_len]
        video_list = video_list[:max_data_len]

    if return_video:
        return data_list, label_list, video_list

    return data_list, label_list


def load_preprocessed_h5_datafile(data_path, indices=None, videos_exist=False):
    """Loads data from a preprocessed .h5 file
    Inputs:
        data_paths: the path to the preprocessed .h5 file
        indices (optional): a list of indices of data to load (load all unless for testing)
        videos_exits: if true, corresponding video exists for each file in the data_path
    Returns:
        data_list: a list of keypoint sequences, each sequence is in shape (sequence length, feature length)
        label_list: labels for keypoint sequences by default 0 without labeling
    """
    data_list = []
    label_list = []
    video_list = []

    f = h5py.File(data_path, 'r')

    if indices is not None:
        for i in indices:
            if np.shape(f[str(i)][:])[0] > 2:
                x = f[str(i)][:]
                y = f['label'][i]

                video = f['videos'][i]

                x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)
                video_list.append(video)

    else:
        # Pre-processed data with multiple videos
        for i in range(len(f['videos'])):

            if np.shape(f[str(i)][:])[0] >= 2:
                x = f[str(i)][:]
                y = f['label'][i]
                x = torch.tensor(x, dtype=torch.float)

                data_list.append(x)
                label_list.append(y)
                if videos_exist:
                    video = f['videos'][i]
                    video_list.append(video)

    if videos_exist:
        return data_list, label_list, video_list

    return data_list, label_list


def downsample_frames(data_list, label_list, video_list, video_dir, video_list_filepath,
                      is_multi_action=True, single_action_crop=50, multi_action_crop=10):
    import cv2
    """ Downsampls or segments keypoint sequences and corresponding video frames into multiple segments as needed
        
    Inputs:
        data_list: list of keypoint sequences
        label_list: list which includes labels for keypoint sequences in the data_list
        video_list: list of video directories for each sequence in the data_list
        video_dir: the directory to save segmented videos
        video_list_filepath: the directory to save the .txt file which contains video directories in video_list
        is_multi_action: true, a keypoint sequence contains multiple behaviors
        singe_action_crop: the maximum length for a sequence with single action
        multi_action_crop: the length of time-window to segment a sequence, used if is_multi_action is true
    Returns:
        data_list_new_2: list of downsampled or segmented keypoint sequences
        label_list_new: list which includes labels for sequences in data_list_new_2
        video_list_new: list which includes video directories for sequences in data_list_new_2
    """

    print(video_list)
    print(data_list)

    if is_multi_action:
        data_list_new_1 = []
        label_list_new = []
        video_list_new = []

        # split video into multiple actions
        for datapoint, label, video in zip(data_list, label_list, list(video_list)):
            print(video)
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            new_datapoints = list(torch.split(datapoint, multi_action_crop))
            data_list_new_1 += new_datapoints
            label_list_new += [label] * len(new_datapoints)

            video_clips = [
                os.path.join(video_dir, "{0}-segment{2}-{3}-{4}.{1}".format(
                    *os.path.basename(video).rsplit('.', 1) + [i] + [multi_action_crop, single_action_crop]))
                for i in range(len(new_datapoints))
            ]

            for i, (new_dp, vid_path) in enumerate(zip(new_datapoints, video_clips)):
                print(vid_path)

                if not os.path.exists(vid_path):
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * multi_action_crop)

                    for i in range(len(new_dp)):
                        ret, frame = cap.read()
                        out.write(frame)

                    out.release()
                    out = None

            video_list_new += video_clips

    else:
        data_list_new_1 = data_list
        label_list_new = label_list
        video_list_new = [os.path.join(video_dir, os.path.basename(video)) for video in video_list]

        for og_path, new_path in zip(video_list, video_list_new):
            shutil.copyfile(og_path, new_path)

    data_list_new_2 = []

    for datapoint, vid_path in zip(data_list_new_1, video_list_new):
        # Downsamples the data because actions can be very long
        if len(datapoint) > single_action_crop:
            indx = np.sort(np.random.randint(0, len(datapoint), single_action_crop))
            datapoint = datapoint[indx]

            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            new_frames = []

            for i in range(num_frames):
                ret, frame = cap.read()

                if i in indx:
                    new_frames.append(frame)

            if os.path.exists(vid_path):
                os.remove(vid_path)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

            for frame in new_frames:
                out.write(frame)

            out.release()

        data_list_new_2.append(datapoint)

    print('DATA LIST LENGTHS', len(data_list), len(data_list_new_2))
    print('DATA POINTS LENGTHS', len(data_list[0]), len(data_list_new_2[0]))

    with open(video_list_filepath, 'w') as video_list_file:
        print(video_list_new)
        video_list_file.writelines('\n'.join(video_list_new))

    return data_list_new_2, label_list_new, video_list_new


def save_compiled_h5_datafile(filepath, data_list: list, label_list, video_list):
    """Saves the organized keypoint sequences into a .h5 file

    Inputs:
        filepath: the directory to save the .h5 file
        data_list: list of keypoint sequences
        label_list: list of labels for sequences in data_list
        video_list: list of video directories for sequences in data_list
    """
    f = h5py.File(filepath, 'w')

    for idx, data in enumerate(data_list):
        dset = f.create_dataset(str(idx), data=np.array(data.numpy()))
    grp = f.create_dataset('label', data=np.array(label_list))

    asciiList = [n.encode("ascii", "ignore") for n in video_list]
    f.create_dataset('videos', (len(asciiList),), 'S200', data=asciiList)

    f.close()


def add_videonames_to_h5(filepath, video_list):
    """Saves video directories to the organized files

    Inputs:
        filepath: a directory to save the organized file
        video_list: a list of video directories for corresponding keypoint sequences
    """
    f = h5py.File(filepath, 'a')

    if 'videos' not in list(f.keys()):
        asciiList = [n.replace('\n', '').encode("ascii", "ignore") for n in video_list]
        f.create_dataset('videos', (len(asciiList),), 'S200', data=asciiList)

    f.close()


def pad_collate_semi(batch):
    """Organizes a batch of samples for pytorch data loader

    Inputs:
        batch: list of instances, each instance includes: (0) a keypoint sequence, (1) the label of the sequence,
        (2) the label indicator of the sequence.
    Returns:
        xx_pad: padded sequence tensor (all samples have the same time length by padding 0).
        lens: lengths for the sequences before padding.
        label: labels for the sequences.
        semi_label: indicators for labeling status of each sequence ( 0 -> unlabeled sequences).
    """
    lens = [len(x[0]) for x in batch]
    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    semi_label = [x[2] for x in batch]
    semi_label = np.asarray(semi_label)
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    return xx_pad, lens, label, semi_label


def pad_collate_iter(batch):
    """Organizes a batch of samples for pytorch data loader.

    Inputs:
        batch: a list of instances, each instance includes: (0) a keypoint sequence, (1) the label of the sequence,
        (2) the semi_label of the sequence

    Returns:
        xx_pad: padded sequence tensor (all samples have the same time length after padding)
        lens: lengths for the sequences before padding.
        label: labels for the sequences.
        semi_label: indicators for labeling status of each sequence ( 0 -> unlabeled sequences).
        index: the unique id of each sequence
    """
    lens = [len(x[0]) for x in batch]
    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    semi_label = [x[2] for x in batch]
    semi_label = np.asarray(semi_label)
    idex = [x[3] for x in batch]
    idex = np.asarray(idex)
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)

    return xx_pad, lens, label, semi_label, idex


class SupDataset(Dataset):
    def __init__(self, root_path, data_path, data_files, label_path=None, test=False, keypoint_names: list = None):
        """Dataloader with the labeled samples

        Inputs:
            root_path: the root directory of the dataset
            data_path: the relative path of the dataset folder
            data_files: the file name of the dataset
            label_path: the directory to save user annotated results
            test: load dataset in testing mode if set to true
            keypoint_names: the names of keypoints 
        """
        data_paths = get_data_paths(root_path, data_path, data_files)
        self.data, self.label = get_data_list(data_paths, keypoint_names=keypoint_names)

        if test:
            # load ground truth label
            self.semi_label = np.ones(len(self.label))
        else:
            if label_path == None:
                self.semi_label = np.zeros(len(self.label))
            else:
                self.semi_label = np.load(label_path)

    def __getitem__(self, index):
        """Gets a sequence according to the unique index in the dataset.

        Inputs:
            index: the id of the sequence to extract.
        Output:
            sequence: the indexed keypoint sequence.
            label: the label of the current sample.
            semi_label: the semi_label of the current sample.
            index: the index of the current sample in the dataset.
        """
        sequence = self.data[index]
        step = sequence.shape[0]
        if step > 40:
            idx = np.random.choice(step, 40)
            idx = np.sort(idx)
            sequence = sequence[idx, :]
        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, label, semi_label, index

    def __len__(self):
        return len(self.label)


class UnsupData(Dataset):
    def __init__(self, data_path, percentage=1, keypoint_names: list = None):
        """Dataloader for the unsupervised training

        Inputs:
            data_path: the directory of a dataset
            keypoint_names: names of keypoints
        """
        self.data, self.label = get_data_list(data_path, keypoint_names=keypoint_names)
        self.semi_label = np.zeros(len(self.label))

    def __getitem__(self, index):
        """Gets a sequence according to unique index in the dataset.

        Inputs:
            index: the id of the sequence to extract.
        Output:
            sequence: a keypoint sequence.
            label: the label of the current sample.
            semi_label: the semi_label of the current sample.
            index: the index of the current sample in the dataset.
        """
        sequence = self.data[index]
        steps = sequence.shape[0]
        if steps > 40:
            idx = np.random.choice(steps, 40)
            idx = np.sort(idx)
            sequence = sequence[idx, :]
        label = self.label[index]
        semi_label = self.semi_label[index]

        return sequence, label, semi_label, index

    def __len__(self):
        return len(self.label)
