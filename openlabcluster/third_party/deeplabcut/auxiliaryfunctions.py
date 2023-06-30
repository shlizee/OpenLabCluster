"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/2472d40a4b1a96130984d9f1bff070f15f5a92a9/deeplabcut/utils/auxiliaryfunctions.py
"""

"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import yaml
import os
from pathlib import Path
from ruamel.yaml import YAML


def create_config_template(multianimal=False):
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    if not multianimal:
        yaml_str = """\
    # Project definitions (do not edit)
        Task:
        \n

    # Project path (change when moving around)
        project_path:
        model_path:
        output_path:
        data_path:
        label_path:
        sample_path:
        train:
        test:
        \n

    # How the video should be preprocessed
    #   - is_single_action: true if videos contain multiple different actions, false otherwise
    #   - multi_action_crop: number of frames to divide videos into
    #   - single_action_crop: number of frames to downsample each video segment into
        is_single_action:
        multi_action_crop:
        single_action_crop:
        \n

    # Training,Evaluation and Analysis configuration
        TrainingFraction:
        iteration:
        default_net_type:
        default_augmenter:
        snapshotindex:
        \n

    # Model Parameters
        feature_length:
        hidden_size:
        en_num_layers:
        de_num_layers:
        cla_num_layers:
        cla_dim:
        num_class:
        teacher_force:
        fix_weigth:
        fix_state:
        \n

    # Training setting
        batch_size:
        learning_rate:
        loss_type:
        un_epoch:
        su_epoch:
        device:
        \n

    # Iter training setting
        sample_per:
        iter_times:
        iter_epoch:
        labeled_id:
        display_iters:
        save_iters:
        multi_epoch:
        tr_modelType:
        tr_modelName:
        sample_method:
        label_budget:
        \n

    # Video Name (Ordered as as in dataset)
        train_videolist:
    # class name the order will be the same as label start from 1 to N
    # make sure to change "cla_dim" and "num_class" if you change number of classes
        class_name:
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile


def read_config(configname):
    """
    Reads a structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                        err.args[2]
                        == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg


def write_config(configname, cfg):
    """
    Writes a structured config file.
    """
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(
            cfg.get("multianimalproject", False)
        )
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if not "skeleton" in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)


def edit_config(configname, edits, output_name=""):
    """
    Edits and saves a config file from a dictionary.

    Parameters
    ----------
    configname : string
        String containing the full path of the config file in the project.
    edits : dict
        Key–value pairs to edit in config
    output_name : string, optional (default='')
        Overwrite the original config.yaml by default.
        If passed in though, new filename of the edited config.

    Examples
    --------
    config_path = 'my_stellar_lab/dlc/config.yaml'

    edits = {'numframes2pick': 5,
             'trainingFraction': [0.5, 0.8],
             'skeleton': [['a', 'b'], ['b', 'c']]}

    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)
    """
    cfg = read_plainconfig(configname)
    for key, value in edits.items():
        cfg[key] = value
    if not output_name:
        output_name = configname
    write_plainconfig(output_name, cfg)
    return cfg


def read_plainconfig(configname):
    """
    Reads a config file.
    """
    if not os.path.exists(configname):
        raise FileNotFoundError(
            f"Config {configname} is not found. Please make sure that the file exists."
        )
    with open(configname) as file:
        return YAML().load(file)


def write_plainconfig(configname, cfg):
    """
    Reads a plain config file.
    """
    with open(configname, "w") as file:
        YAML().dump(cfg, file)
