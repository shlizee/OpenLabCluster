"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/gui/train_network.py
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
import pydoc
import sys
import wx


class Train_network(wx.lib.scrolledpanel.ScrolledPanel):
    """
    Defines the parent class for Cluster Map panel in OpenLabCluster based on the original Train network panel of DeepLabCut.
    """

    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent=parent)
        displays = (
            wx.Display(i) for i in range(wx.Display.GetCount())
        )  # Gets the number of displays.
        screenSizes = [
            display.GetGeometry().GetSize() for display in displays
        ]  # Gets the size of each display.
        index = 0  # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth * 0.7, screenHeight * 0.85)
        # Variable initialization
        self.method = "automatic"
        self.config = cfg
        # Design the panel
        self.parent = parent
        self.sizer = wx.GridBagSizer(6, 5)
        subsizer = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(self, label="OpenLabCluster - Step 2. Generate Cluster Map")
        subsizer.Add(text, 0, wx.EXPAND)
        self.sizer.Add(subsizer, pos=(0, 0), span=(1, 5), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.BOTTOM, border=15)
        line1 = wx.StaticLine(self)
        self.sizer.Add(
            line1, pos=(1, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=5
        )

    def help_function(self, event):
        """
        Defines help function.
        """

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.train_iter_network"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

    def train_network(self, event):
        """
        Initializes the training function.
        """
        if self.shuffles.Children:
            shuffle = int(self.shuffles.Children[0].GetValue())
        else:
            shuffle = int(self.shuffles.GetValue())

        if self.trainingindex.Children:
            trainingsetindex = int(self.trainingindex.Children[0].GetValue())
        else:
            trainingsetindex = int(self.trainingindex.GetValue())

        if self.snapshots.Children:
            max_snapshots_to_keep = int(self.snapshots.Children[0].GetValue())
        else:
            max_snapshots_to_keep = int(self.snapshots.GetValue())

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

    def cancel_train_network(self, event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.pose_cfg_text.Hide()
        self.update_params_text.Hide()
        self.pose_cfg_choice.SetSelection(1)
        self.display_iters.SetValue(100)
        self.save_iters.SetValue(10000)
        self.max_iters.SetValue(50000)
        self.snapshots.SetValue(5)
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
