"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li and Moishe Keselman.
"""

import os
import numpy as np
import traceback

import wx
import wx.lib.scrolledpanel

# Third party import
from openlabcluster.third_party.deeplabcut.create_new_project import Create_new_project
from openlabcluster.third_party.deeplabcut import new, auxiliaryfunctions

# OpenLabCluster function import
import openlabcluster
from openlabcluster.user_interface.cluster_map import Cluster_map
from openlabcluster.user_interface.classify_behavior import selection_method_options
from openlabcluster.training_utils.ssl.data_loader import DataFileLoadingError, FeatureNameKeyError, \
    add_videonames_to_h5, downsample_frames, get_data_list, get_data_paths, save_compiled_h5_datafile

class Project_constructor(Create_new_project):
    """
    This class defines the Manage Project page
    Notes:
    The class inherits from the Create_new_project class in third_party 
    """
    def __init__(self, parent, gui_size, statusBar):
        """
        Constructs the Manage Project page
        Inputs:
            parent: parent panel
            gui_size: the height and the width of the GUI window
            statusBar: the handle for statusBar
        """
        super(Project_constructor, self).__init__(parent, gui_size)
        self.statusBar = statusBar
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER, size=(gui_size[0], gui_size[1]))
        # variable initialization
        self.traininglist = None
        self.testinglist = None
        self.trainvideo_list = None
        self.testvideo_list = None
        self.design_panel_top()

        # Defines panels to manage project.
        self.trds = wx.StaticText(self, label="Keypoints Data (Launch DeepLabCut if Unavailable):")
        self.sizer.Add(self.trds, pos=(4, 0), span=(1, 2), flag=wx.TOP | wx.LEFT, border=10)

        self.sel_pp_data_label = "Load Preprocessed Keypoints"
        self.sel_pp_data = wx.Button(self, label=self.sel_pp_data_label)
        self.sizer.Add(self.sel_pp_data, pos=(4, 2), span=(1, 2), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_pp_data.Bind(wx.EVT_BUTTON, self.select_training_data)

        self.sel_trds_label = "Load Keypoints (h5 file)"
        self.sel_trds = wx.Button(self, label=self.sel_trds_label)
        self.sizer.Add(self.sel_trds, pos=(4, 4), span=(1, 2), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_trds.Bind(wx.EVT_BUTTON, self.select_training_data)

        self.launch_deeplabcut_btn = wx.Button(self, label="Launch DeepLabCut")
        self.sizer.Add(self.launch_deeplabcut_btn, pos=(4, 6), span=(1, 1.5), flag=wx.TOP | wx.EXPAND, border=5)
        self.launch_deeplabcut_btn.Bind(wx.EVT_BUTTON, self.launch_deeplabcut)

        self.sel_data_btn_list = [self.sel_trds, self.sel_pp_data]
        self.sel_data_btn_lbl_list = [self.sel_trds_label, self.sel_pp_data_label]

        self.trvl = wx.StaticText(self, label="Video Segments Names List:")
        self.sizer.Add(self.trvl, pos=(5, 0), span=(1, 2), flag=wx.TOP | wx.LEFT, border=10)

        self.sel_trvl = wx.Button(self, label="Load Video Segments Names List")
        self.sizer.Add(
            self.sel_trvl, pos=(5, 2), span=(1, 5), flag=wx.TOP | wx.EXPAND, border=10
        )
        self.sel_trvl.Bind(wx.EVT_BUTTON, self.load_training_videoname_list)
        self.design_panel_middle()
        self.advanced_options = wx.CollapsiblePane(self, wx.ID_ANY, "Advanced Options")
        self.advanced_options_sizer = wx.GridBagSizer(8, 8)
        self.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.on_advanced_options_change)
        advanced_options_win = self.advanced_options.GetPane()

        select_text = wx.StaticBox(advanced_options_win, label="Active Learning Method")
        selectboxsizer = wx.StaticBoxSizer(select_text, wx.VERTICAL)
        self.select_choice = wx.ComboBox(advanced_options_win, style=wx.CB_READONLY)
        self.select_choice.Set(selection_method_options)
        self.select_choice.SetValue(selection_method_options[0])
        selectboxsizer.Add(self.select_choice)

        self.advanced_options_sizer.Add(selectboxsizer, pos=(0, 0))
        advanced_options_win.SetSizer(self.advanced_options_sizer)
        self.sizer.Add(self.advanced_options, pos=(9, 0))

        # Starts the project and goes to unsupervised clustering step when click "OK"
        self.ok = wx.Button(self, label="OK")
        self.sizer.Add(self.ok, pos=(9, 5))
        self.ok.Bind(wx.EVT_BUTTON, self.create_project_function)
        self.design_panel_bottom()
        self.SetupScrolling()
        self.SetAutoLayout(1)

        self.Scroll(0, 0)

    def launch_deeplabcut(self, event):
        """
        Launches DeepLabCut to extract body keypoints if needed
        """
        from sys import platform
        if platform == "linux" or platform == "linux2":
            # linux
            command = 'python -m deeplabcut'
        elif platform == "darwin":
            # OS X
            command = 'pythonw -m deeplabcut'
        elif platform == "win32":
            # Windows...
            command = 'python -m deeplabcut'
        # Opens DeepLabCut as the subprocess.
        import subprocess
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    def help_function(self, event):
        """
        Defines the help window
        """
        help_text = """Documentation:
        1. Project Name - name for the project
        2. If you have videos only, you would need to use markerless pose estimators (e.g. DeepLabCut) to extract keypoints. If you already have h5 formatted files (as output of DeepLabCut), select them with "Load Keypoints Data".
        3. Select your videos with "Choose Training Videos List". There should be one video for each keypoint file. Make sure the videos and keypoints files are of the same order.
        4. Optional: Set a directory for the project (the default is the working directory of this project).
        5. Keep the GPU box checked if you have a GPU on your computer and you would like to use it for training.
        6. Enter the features length (number of body parts * number of dimensions per body part. For example, for 5 keypoints in 2D, this would be 5*2 =10).
        7. Choose "OK" to create the project.
        8. If you would like to edit the config (i.e. change class names, change video cropping), press "Edit Config File"."""
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)

    def on_advanced_options_change(self, event):
        self.FitInside()

    def chooseOption(self, event):
        def show_hide_load_project(should_show: bool):
            self.proj_name_txt_box.Enable(should_show)
            self.proj_name.Enable(should_show)
            self.trds.Enable(should_show)
            self.trvl.Enable(should_show)
            self.sel_trvl.Enable(should_show)
            self.sel_trds.Enable(should_show)
            self.sel_pp_data.Enable(should_show)
            self.change_workingdir.Enable(should_show)
            self.launch_deeplabcut_btn.Enable(should_show)
            self.hardware_type.Enable(should_show)
            self.feature_length.Enable(should_show)
            self.feature_length_txt_box.Enable(should_show)
            self.sel_wd_text.Enable(should_show)

        def show_hide_config(should_show: bool):
            if should_show:
                self.sel_config.Show()
                self.cfg_text.Show()
            else:
                if self.sel_config.IsShown():
                    self.sel_config.Hide()
                    self.cfg_text.Hide()

        if self.proj.GetStringSelection() == self.load_project_rbo:

            if self.loaded:
                self.sel_config.SetPath(self.cfg)

            show_hide_load_project(False)
            show_hide_config(True)

            self.sizer.Fit(self)
        else:
            show_hide_load_project(True)
            show_hide_config(False)

            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

        self.parent.Parent.Layout()


    def select_training_data(self, event: wx.Event):
        """
        Selects the directory for dataset
        """
        cwd = os.getcwd()
        tdlg = wx.FileDialog(
            self, "Select data for training", cwd, "", "*.*", wx.FD_MULTIPLE
        )
        if tdlg.ShowModal() == wx.ID_OK:
            self.traininglist = tdlg.GetPaths()

            for btn, lbl in zip(self.sel_data_btn_list, self.sel_data_btn_lbl_list):
                btn.SetLabel(lbl)

            event.GetEventObject().SetLabel(f"{len(self.traininglist)} files selected")

            if event.GetEventObject() is self.sel_pp_data:
                self.traininglist = self.traininglist[0]

    def load_training_videoname_list(self, event):
        """
        Selects the directory of the .txt file containing video directories of keypoint sequences
        """
        cwd = os.getcwd()
        dlg = wx.FileDialog(
            self, "Select training video list", cwd, "", "*.*", wx.FD_MULTIPLE
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.trainvideo_list = dlg.GetPaths()
            self.sel_trvl.SetLabel(f"{len(self.trainvideo_list)} videos selected")


    def create_project_function(self, event):
        """
        Creates a new project
        """
        if self.sel_config.IsShown():
            self.cfg = self.sel_config.GetPath()
            if self.cfg == "":
                wx.MessageBox(
                    "Please choose the config.yaml file to load the project",
                    "Error",
                    wx.OK | wx.ICON_ERROR,
                )
                self.loaded = False
            else:
                self.statusBar.SetStatusText("Project Loaded! Wait for Loading Model")
                self.loaded = True
                self.edit_config_file.Enable(True)
        else:
            # Loads previous project.
            self.task = self.proj_name_txt_box.GetValue()

            if self.task != "" and self.traininglist != None and self.trainvideo_list != None:
                self.cfg = new.create_new_project(
                    self.task,
                    self.traininglist,
                    self.trainvideo_list,
                    self.dir,
                    working_directory=self.dir,
                    copy_videos=self.copy,
                    use_gpu=self.hardware_type.GetValue(),
                    feature_length=int(self.feature_length_txt_box.GetValue()),
                    sample_method=self.select_choice.GetStringSelection(),
                )
            else:
                wx.MessageBox(
                    "Some of the enteries are missing.\n\nMake sure that the task and experimenter name are specified"
                    "and training data are selected!",
                    "Error",
                    wx.OK | wx.ICON_ERROR,
                )
                self.cfg = False
            if self.cfg:
                self.statusBar.SetStatusText("New Project Created")
                self.loaded = True
                self.edit_config_file.Enable(True)

        if self.parent.GetPageCount() > 3:
            for i in range(2, self.parent.GetPageCount()):
                self.parent.RemovePage(2)
                self.parent.Layout()

        # Adds all the other pages
        if self.loaded:
            self.edit_config_file.Enable(True)
            cfg = auxiliaryfunctions.read_config(self.cfg)

            # Loads dataset.
            train_files = 'train_files'

            if cfg.get(train_files) is not None:
                paths = get_data_paths(cfg["project_path"], cfg['data_path'], cfg[train_files])

                try:
                    data_list, label_list, datafile_list = get_data_list(paths, keypoint_names=cfg.get('feature_names'),
                                                                         return_video=True)

                    if (len(data_list) == 0):
                        # If dataset doesn't exist or its empty.
                        wx.MessageBox("Empty dataset", 'Error', wx.OK | wx.ICON_ERROR)
                        return
                except FeatureNameKeyError as e:
                    wx.MessageBox("Following feature_name doesn't exist in the dataset:\n" + str(
                        e) + "\n\nPlease check your dataset and/or your config", 'Error', wx.OK | wx.ICON_ERROR)
                    return
                except DataFileLoadingError as e:
                    # If fails to load a data file.
                    wx.MessageBox(
                        "File failed to load: \n" + str(e) + "\n\nPlease check that it exists and it can be loaded",
                        'Error', wx.OK | wx.ICON_ERROR)
                    return
                # Cuts the video into small segments.
                # If multi_action_crop is True, crops the video into sub-segments.
                # If single_action_crop is True, downsamples the video.
                data_list, label_list, datafile_list = downsample_frames(data_list, label_list, cfg['train_videos'],
                                                                         cfg['video_path'], cfg['train_videolist'],
                                                                         is_multi_action=not cfg['is_single_action'],
                                                                         single_action_crop=cfg['single_action_crop'],
                                                                         multi_action_crop=cfg['multi_action_crop']
                                                                         )

                save_compiled_h5_datafile(os.path.join(cfg['data_path'], cfg['train']), data_list, label_list,
                                          datafile_list)
            else:
                # Saves to h5 file.
                if cfg['train_videolist'].endswith('.npy'):
                    frame_names = np.load(cfg['train_videolist'])
                    add_videonames_to_h5(os.path.join(cfg['data_path'], cfg['train']), frame_names[:, 0].tolist())
                else:
                    with open(cfg['train_videolist'], 'r') as f:
                        video_list = f.readlines()
                        add_videonames_to_h5(os.path.join(cfg['project_path'], cfg['data_path'], cfg['train']),
                                             video_list)

            if self.parent.GetPageCount() < 3:
                # No clustering page is set up.
                try:
                    # Adds the page to train the cluster network.
                    page6 = Cluster_map(self.parent, self.gui_size, self.cfg, self.statusBar)
                except Exception as e:
                    wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_ERROR)
                    print(traceback.format_exc())
                    return

                self.parent.AddPage(page6, "Cluster Map")
                self.statusBar.SetStatusText('Model Loaded. Go to Cluster Map')
                self.edit_config_file.Enable(True)
