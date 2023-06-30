"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li and Moishe Keselman.
"""

import os
import traceback
import numpy as np
import wx
from wx.lib.pubsub import pub
from wx.lib.scrolledpanel import ScrolledPanel

# Third party import
from openlabcluster.third_party.deeplabcut.train_network import Train_network
from openlabcluster.third_party.deeplabcut.widgets import ImagePanel
from openlabcluster.third_party.deeplabcut import auxiliaryfunctions

# OpenLabCluster import
import openlabcluster
from openlabcluster.user_interface.extract_hidden import transform_hidden
from openlabcluster.user_interface.active_labeling import video_display_window
from openlabcluster.user_interface.plotting_utils import format_axes

media_path = os.path.join(openlabcluster.__path__[0], "user_interface", "media")

class Cluster_map(Train_network):
    """
    Initializes the Cluster Map panel
    Notes:
    The class inherits from class Train_network in third_party
    """

    def __init__(self, parent, gui_size, cfg, statusBar):
        """
        Constructs the Behavior Clustering Map panel
        Inputs:
            parent: parent panel
            gui_size: the height and the width of the GUI window
            cfg: the config file directory
            statusBar: the handle for statusBar
        """
        super(Cluster_map, self).__init__(parent, gui_size, cfg)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        pose_cfg = auxiliaryfunctions.read_config(self.config)
        display_iters = str(pose_cfg["display_iters"])
        save_iters = str(pose_cfg["save_iters"])
        max_iters = str(pose_cfg["su_epoch"])

        self.sizer.AddGrowableCol(2)
        self.statusBar = statusBar
        self.initial = True
        self.oper_1 = wx.BoxSizer(wx.HORIZONTAL)

        # Initializes image and video panel.
        self.image_panel = ImagePanel(self)
        self.video_display = video_display_window(self, self.gui_size, pose_cfg)
        self.sizer.Add(self.image_panel, pos=(4,0), span=(1,2))
        self.sizer.Add(self.video_display, pos=(4, 2), span=(1, 1))
        scroll_panel_sizer = wx.BoxSizer( wx.VERTICAL )
        scroll_panel_sizer.Add(self.image_panel)

        # Initialized default parameters.
        self.scroll_panel = ScrolledPanel(self)
        self.scroll_panel.SetSizer(scroll_panel_sizer)
        display_iters_text = wx.StaticBox(self, label="Update Cluster Map Every (Epochs)")
        display_iters_text_boxsizer = wx.StaticBoxSizer(display_iters_text, wx.VERTICAL)
        self.display_iters = wx.SpinCtrl(
            self, value=display_iters, min=1, max=1000
        )
        display_iters_text_boxsizer.Add(
            self.display_iters, 1, wx.EXPAND |wx.TOP |wx.BOTTOM, 5
        )

        save_iters_text = wx.StaticBox(self, label="Save Cluster Map Every (Epochs)")
        save_iters_text_boxsizer = wx.StaticBoxSizer(save_iters_text, wx.VERTICAL)
        self.save_iters = wx.SpinCtrl(self, value=save_iters, min=1, max=1000)
        save_iters_text_boxsizer.Add(
            self.save_iters, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 5
        )

        max_iters_text = wx.StaticBox(self, label="Maximum Epochs")
        max_iters_text_boxsizer = wx.StaticBoxSizer(max_iters_text, wx.VERTICAL)

        self.max_iters = wx.SpinCtrl(self, value=max_iters, min=1, max=1000)
        max_iters_text_boxsizer.Add(
            self.max_iters, 1, wx.EXPAND | wx.TOP| wx.BOTTOM, 5
        )

        dim_text = wx.StaticBox(self, label="Cluster Map Dimension")
        dimboxsizer = wx.StaticBoxSizer(dim_text, wx.VERTICAL)
        self.dim_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        self.dim_choice.Bind(wx.EVT_COMBOBOX, self.update_image_panel)
        options = ["2d", "3d"]
        self.dim_choice.Set(options)
        self.dim_choice.SetValue("2d")
        dimboxsizer.Add(self.dim_choice, 10, wx.EXPAND  |wx.TOP | wx.BOTTOM, 5)
        reducer_text = wx.StaticBox(self, label="Dimension Reduction Method" )
        reducerboxsizer = wx.StaticBoxSizer(reducer_text, wx.VERTICAL)
        self.reducer_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        self.reducer_choice.Bind(wx.EVT_COMBOBOX, self.update_image_panel)
        reducer_options = ["PCA", "tSNE", "UMAP"]
        self.reducer_choice.Set(reducer_options)
        self.reducer_choice.SetValue("PCA")
        reducerboxsizer.Add(self.reducer_choice, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        hbox2.Add(display_iters_text_boxsizer, 10, wx.EXPAND | wx.BOTTOM, 5)
        hbox2.Add(save_iters_text_boxsizer, 10, wx.EXPAND  | wx.BOTTOM, 5)
        hbox2.Add(max_iters_text_boxsizer, 10, wx.EXPAND  | wx.BOTTOM, 5)
        hbox2.Add(dimboxsizer,10, wx.EXPAND | wx.BOTTOM, 5)
        hbox2.Add(reducerboxsizer, 10, wx.EXPAND| wx.BOTTOM, 5)

        self.sizer.Add(
            hbox2,
            pos=(2, 0),
            span=(1, 5),
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT |wx.TOP | wx.BOTTOM,
            border=5,
        )
        
        # Clustering stage buttons.
        botton_box = wx.BoxSizer(wx.HORIZONTAL)
        self.ok = wx.Button(self, label="Start Clustering")
        botton_box.Add(self.ok, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.ok.Bind(wx.EVT_BUTTON, self.train_network)

        self.stop = wx.Button(self, label="Stop Clustering")
        self.stop.Bind(wx.EVT_BUTTON, self.stop_train)
        botton_box.Add(self.stop, 10, wx.EXPAND  | wx.TOP | wx.BOTTOM, 5)

        self.continue_but = wx.Button(self, label="Continue Clustering")
        self.continue_but.Bind(wx.EVT_BUTTON, self.continue_train)
        botton_box.Add(self.continue_but, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        self.action = wx.Button(self, label="Go to Classification")
        self.action.Bind(wx.EVT_BUTTON, self.action_recognition_panel)
        botton_box.Add(self.action, 10, wx.EXPAND | wx.BOTTOM, 5)

        self.help_button = wx.Button(self, label="Help")
        botton_box.Add(self.help_button, 10, wx.EXPAND  | wx.BOTTOM, 5)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)
        self.sizer.Add(botton_box, pos=(3, 0),
                       span=(1, 5),
                       flag=wx.EXPAND |  wx.LEFT | wx.RIGHT,
                       border=10, )

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        self.selected_id = []

        # Initializes hidden state plot.
        self.work = transform_hidden(self.config, reducer_name=self.reducer_choice.GetValue())
        transform = self.work.transformed()
        self.sc = self.image_panel.axes.scatter(transform[:, 0], transform[:,1], s=10, picker=True, color='k')
        self.annot = self.image_panel.axes.annotate("", xy=(0, 0), xytext=(5, 5), textcoords="offset points")
        self.annot.set_visible(False)

        self.image_panel.canvas.mpl_connect('pick_event', self.display_data)
        self.image_panel.canvas.mpl_connect("motion_notify_event", self.hover)
        self.image_panel.axes.set_xlim([-10, 20])
        self.image_panel.axes.set_ylim([-10, 20])

        if type(pose_cfg['train_videolist']) is str:
            if pose_cfg['train_videolist'].endswith('.text') or pose_cfg['train_videolist'].endswith('.txt'):
                with open(pose_cfg['train_videolist'], 'r') as f:
                    self.videpaths = f.readlines()
            elif pose_cfg['train_videolist'].endswith('.npy'):
                self.videpaths = np.load(pose_cfg['train_videolist'], allow_pickle=True)

        self.image_panel.axes.set_title('Cluster Map')

        format_axes(self.image_panel.axes)
        pub.subscribe(self.on_finish, "finish")

        self.SetupScrolling()


    def OnMouseLeave(self, event):
        """
        Defines the on-leave mouse event
        """
        self.statusBar.SetStatusText("")


    def help_function(self, event):
        """
        Defines the help window
        """

        help_text = """
       Set the Training Parameters:
        1. Update Cluster Map Every (Epochs): this helps you to decide when to update the Cluster Map, e.g. set 1 to update Cluster Map every training epoch, set to 5, update every 5 epoch
        2. Save Cluster Map Every (Epochs): this decides when to save Cluster Maps, e.g. if it is 1, update the Cluster Map every training epoch
        3. Maximum Epochs: the number of epochs perform training
        4. Cluster Map Dimension: you can choose "2d" or "3d", if it is "2d" the Cluster Map will be shown in 2D dimension, otherwise it is 3D dimension
        5. Dimension Reduction Methods: possible choices are "PCA", "tSNE", "UMAP". The GUI will use the chosen method to perform dimension reduction and show results in the ClusterMap
        
        
        Buttons:
        After setting the parameters you can perform analysis:
        
        1. Start Clustering: perform an unsupervised sequence regeneration task
        3. Stop Clustering: usually, the clustering will stop when it reaches the maximum epochs, but if you want to stop at an intermediate stage, click this button
        2. Continue Clustering: if you stopped the clustering at some stage and want to perform clustering with earlier clustering results, click this button
        4. Reset: Reset the earlier defined training parameters to default
        5. Go to Classification: after the unsupervised clustering we go to the next step which includes: i) annotation suggestion, ii) sample annotating and iii) semi-supervised action classification with labeled samples

        """
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)


    def draw_clustermap(self):
        """
        Updates the ClusterMap plot
        """
        self.work.plot()


    def train_network(self, event):
        """
        Starts the model training
        """
        pub.subscribe(self.on_finish, "finish")
        pub.subscribe(self.draw_clustermap, "plot")
        if self.display_iters.Children:
            displayiters = int(self.display_iters.Children[0].GetValue())
        else:
            displayiters = int(self.display_iters.GetValue())

        if self.save_iters.Children:
            saveiters = int(self.save_iters.Children[0].GetValue())
        else:
            saveiters = int(self.save_iters.GetValue())

        if self.max_iters.Children:
            maxiters = int(self.max_iters.Children[0].GetValue())
        else:
            maxiters = int(self.max_iters.GetValue())
        try:
            self.work=openlabcluster.train_unsup_network(
                self.config,
                self.image_panel,
                displayiters=displayiters,
                saveiters=saveiters,
                maxiters=maxiters,
                reducer_name=self.reducer_choice.GetValue(),
                dimension= self.dim_choice.GetValue()
            )
            self.statusBar.SetStatusText('Training Started')
            self.ok.Enable(False)
            self.continue_but.Enable(False)
            self.dim_choice.Enable(False)
            self.reducer_choice.Enable(False)
            self.max_iters.Enable(False)
            self.display_iters.Enable(False)
            self.save_iters.Enable(False)
            self.action.Enable(False)
        except Exception as e:
            print(traceback.format_exc())
            wx.MessageBox('Error while training. Look in terminal for more information', 'Training Error', wx.OK | wx.ICON_ERROR) 


    def stop_train(self, event):
        """
        Stops training of model
        """
        self.work.stop()
        self.ok.Enable(True)
        self.continue_but.Enable(True)
        self.ok.Enable(True)
        self.continue_but.Enable(True)
        self.dim_choice.Enable(True)
        self.reducer_choice.Enable(True)
        self.max_iters.Enable(True)
        self.display_iters.Enable(True)
        self.save_iters.Enable(True)
        self.action.Enable(True)
        self.statusBar.SetStatusText('Training Stopped')
        self.work.join()
        self.work.plot()
        pub.unsubscribe(self.on_finish, "finish")
        pub.unsubscribe(self.draw_clustermap, "plot")

    def continue_train(self, event):
        """
        Continues training of model from an earlier saved checkpoint
        """
        pub.subscribe(self.on_finish, "finish")
        pub.subscribe(self.draw_clustermap, "plot")
        if self.display_iters.Children:
            displayiters = int(self.display_iters.Children[0].GetValue())
        else:
            displayiters = int(self.display_iters.GetValue())

        if self.save_iters.Children:
            saveiters = int(self.save_iters.Children[0].GetValue())
        else:
            saveiters = int(self.save_iters.GetValue())

        if self.max_iters.Children:
            maxiters = int(self.max_iters.Children[0].GetValue())
        else:
            maxiters = int(self.max_iters.GetValue())
        self.ok.Enable(False)

        try:
            self.work=openlabcluster.train_unsup_network(
                self.config,
                self.image_panel,
                displayiters=displayiters,
                saveiters=saveiters,
                maxiters=maxiters,
                continue_training=True,
                reducer_name=self.reducer_choice.GetValue(),
                dimension=self.dim_choice.GetValue()
            )
            self.statusBar.SetStatusText('Training Continued')
            self.ok.Enable(False)
            self.continue_but.Enable(False)
            self.dim_choice.Enable(False)
            self.reducer_choice.Enable(False)
            self.max_iters.Enable(False)
            self.display_iters.Enable(False)
            self.save_iters.Enable(False)
            self.action.Enable(False)
        except Exception as e:
            print(traceback.format_exc())
            wx.MessageBox('Error while training. Look in terminal for more information', 'Training Error', wx.OK | wx.ICON_ERROR)


    def update_image_panel(self, event):
        """
        Updates the cluster map plot
        """
        dim = self.dim_choice.GetValue()
        self.image_panel.refresh(dim)
        method = self.reducer_choice.GetValue()
        self.work = transform_hidden(self.config, reducer_name = method, dimension=dim)
        transform = self.work.transformed()
        if dim =='2d':
            self.sc = self.image_panel.axes.scatter(transform[:,0], transform[:,1],s=10, picker=True, color='k')
        else:
            self.sc = self.image_panel.axes.scatter(transform[:,0], transform[:,1],transform[:,2], s=10, picker=True, color='k')
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 16,
                }
        self.image_panel.axes.set_title('Cluster Map', fontdict=font)
        self.image_panel.axes.autoscale()
        format_axes(self.image_panel.axes)


    def on_finish(self):
        """
        Stops training and enables a set of buttons
        """
        pub.unsubscribe(self.on_finish, "finish")
        pub.subscribe(self.draw_clustermap, "plot")
        self.ok.Enable(True)
        self.continue_but.Enable(True)
        self.dim_choice.Enable(True)
        self.reducer_choice.Enable(True)
        self.max_iters.Enable(True)
        self.display_iters.Enable(True)
        self.save_iters.Enable(True)
        self.continue_but.Enable(True)
        self.action.Enable(True)
        self.work.join()
        self.work.plot()
        pass


    def action_recognition_panel(self, event):
        """
        Initializes the Behavior Classification Map panel
        """
        from openlabcluster.user_interface.classify_behavior import Behavior_classification
        if self.parent.GetPageCount() < 4:
            page5 = Behavior_classification(self.parent, self.gui_size, self.config, self.statusBar)
            self.parent.AddPage(page5, "Behavior Classification Map")
            self.statusBar.SetStatusText('Go to Behavior Classification Map')

    def display_data(self, event):
        """
        Displays a video selected in the ClusterMap
        """
        thisline = event.artist
        inarray = np.asarray(event.ind)
        print(inarray)
        if len(inarray) > 0:
            ind = inarray[0]
        else:
            ind = inarray
        self.video_display.load_video(ind)


    def hover(self, event):
        """
        Shows the information of a sample selected
        """
        if event.inaxes == self.image_panel.axes:
            # Gets the points contained in the event
            cont, ind = self.sc.contains(event)
            print(ind)
            if cont:
                # Changes annotation position
                self.annot.xy = (event.xdata, event.ydata)
                # Writes the name of every point contained in the event
                if type(self.videpaths[0]) == list:
                    self.statusBar.SetStatusText("{}".format(','.join([f'ID {n}, Path {self.videpaths[n][0]}' for n in ind["ind"]])))
                else:
                    self.statusBar.SetStatusText("{}".format(','.join([f'ID {n}, Path {self.videpaths[n]}' for n in ind["ind"]])))
            else:
                self.statusBar.SetStatusText('')