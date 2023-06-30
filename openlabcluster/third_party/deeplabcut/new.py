"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/create_project/new.py
"""

"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
import shutil
import warnings
from pathlib import Path
from openlabcluster import DEBUG
import wx

def create_new_project(
    project,
    data_name, 
    training_video_list, 
    test_videodir,
    working_directory=None,
    copy_videos=False,
    videotype=".avi",
    multianimal=False,
    use_gpu=True,
    feature_length=None,
    **kwargs
):
    """Creates a new project directory, sub-directories and a basic configuration file. The configuration file is
    loaded with the default values. Change its parameters to your projects need.

    The original function from DeepLabCut is partially revised for the use of OpenLabCluster,
    including text description and page arrangement.

    Parameters
    ----------
    project : string
        String containing the name of the project.

    data_name : list | string
        List of deeplabcut h5 filepath strings or filepath of preprocessed data

    training_video_list : list | string
        List of video filepath string or text/npy filepath of list of video filenames

    **kwargs : additional config items 

    """
    from datetime import datetime as dt
    from openlabcluster.third_party.deeplabcut import auxiliaryfunctions

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory == None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{date}".format(pn=project, date=date)
    project_path = wd / project_name

    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        wx.MessageBox(
            'Project "{}" already exists!'.format(project_path),
            "Error",
            wx.OK | wx.ICON_ERROR,
        )
        return
    video_path = project_path/ "videos"
    label_path = project_path/ "label"
    data_path = project_path/ "datasets"
    model_path = project_path/ "models"
    output_path = project_path/ 'output'
    sample_path = project_path / 'sample'
    for p in [video_path, label_path, data_path, model_path, output_path, sample_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    if not len(data_name):
        # Silently sweep the files that were already written.
        shutil.rmtree(project_path, ignore_errors=True)
        warnings.warn(
            "No valid videos were found. The project was not created... "
            "Verify the video files and re-create the project."
        )
        return "nothingcreated"
    elif type(data_name) is list :
        data_name = [Path(dp) for dp in data_name]
        dest_dp = [data_path.joinpath(dp.name) for dp in data_name]
        for src, dst in zip(data_name, dest_dp):
            shutil.copy(
                os.fspath(src), os.fspath(dst)
            )
            # Sets values to config file:

        data_name = dest_dp

    cfg_file, ruamelFile = auxiliaryfunctions.create_config_template()
    
    # Common parameters:
    cfg_file["Task"] = project
    cfg_file["project_path"] = str(project_path)
    cfg_file['model_path'] = str(model_path)
    cfg_file['tr_modelType'] = 'semi_seq2seq'
    cfg_file['output_path'] = str(output_path)
    cfg_file['data_path'] = str(data_path)
    cfg_file['label_path'] = str(label_path)
    
    if type(data_name) is list:
        cfg_file["train"] = 'compiled.h5'
        cfg_file["train_files"] = [str(name) for name in data_name]
    else:
        cfg_file['train'] = Path(data_name).name
        cfg_file['train_files'] = None
        shutil.copy(
            data_name, 
            os.path.join(data_path, Path(data_name).name)
            )

    cfg_file['sample_path'] = str(sample_path)
    cfg_file['video_path'] = str(video_path)

    cfg_file['is_single_action'] = False
    cfg_file['single_action_crop'] = 10
    cfg_file['multi_action_crop'] = 10

    if len(training_video_list)==1 and any([training_video_list[0].endswith(ending) for ending in ['.txt', '.text', '.npy']]):
        # File with video names is given.
        cfg_file['train_videolist'] = os.path.join(video_path, Path(training_video_list[0]).name)
        cfg_file['train_videos'] = None
        shutil.copy(
            training_video_list[0],
            cfg_file['train_videolist']
        )
    else:
        # List of video files is given
        cfg_file['train_videolist'] = os.path.join(video_path, 'videos.txt')
        cfg_file['train_videos'] = training_video_list

    if test_videodir:
        cfg_file['test_videolist'] = test_videodir

    cfg_file['feature_length']= (16 if feature_length is None else feature_length)
    cfg_file['hidden_size']= 215
    cfg_file['batch_size']= 64
    cfg_file['en_num_layers']= 3
    cfg_file['de_num_layers']= 1
    cfg_file['cla_num_layers']= 1
    cfg_file['learning_rate']= 0.0001
    cfg_file['un_epoch']= 30
    cfg_file['cla_dim']= [8]
    cfg_file['num_class'] = [8]
    cfg_file['su_epoch'] = 30
    cfg_file['loss_type'] = 'L1'
    cfg_file['sample_method'] = 'Cluster Center'
    cfg_file['teacher_force']=False
    cfg_file['fix_weight']= False
    cfg_file['fix_state']= False
    cfg_file['device'] = ('cuda' if use_gpu else 'cpu')
    cfg_file["display_iters"] = 3
    cfg_file["save_iters"]= 30
    cfg_file["multi_epoch"] = 30
    cfg_file['class_name'] = [ 'drink', 'eat', 'groom', 'hang', 'head', 'rear', 'rest', 'walk']
    cfg_file['label_budget'] = 10

    for key, value in kwargs.items():
        cfg_file[key] = value

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile, cfg_file)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        "\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)."
        % (project_name, str(wd))
    )
    return projconfigfile
