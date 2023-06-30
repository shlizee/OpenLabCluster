"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/gui/create_new_project.py
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
import platform
import pydoc
import subprocess
import sys
import webbrowser
import wx

class Create_new_project(wx.lib.scrolledpanel.ScrolledPanel):
    """
    Creates Manage Project page.
    """
    """
    The original class is partially revised for the use of OpenLabCluster, 
    including text description and page arrangement changes. 
    """

    def __init__(self, parent, gui_size):
        self.gui_size = gui_size
        self.parent = parent
        h = gui_size[0]
        w = gui_size[1]
        # variable initialization
        self.filelist = []
        self.filelistnew = []
        self.dir = None
        self.copy = False
        self.cfg = None
        self.loaded = False

    def design_panel_top(self):
        """
        Creates top part of the Manage Project page, revised for the use of OpenLabCluster.
        """
        self.sizer = wx.GridBagSizer(14, 15)
        subsizer = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(self, label="OpenLabCluster - Step 1. Create a New Project or Load a Project")
        subsizer.Add(text, 0, wx.EXPAND)

        self.sizer.Add(subsizer, pos=(0, 0), span=(1, 15), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.BOTTOM, border=15)

        line = wx.StaticLine(self)
        self.sizer.Add(
            line, pos=(1, 0), span=(1, 15), flag=wx.EXPAND | wx.BOTTOM, border=5
        )

        # Stores radio box options
        self.new_project_rbo = "New Project"
        self.load_project_rbo = "Load Project"

        # Adds all the options
        self.proj = wx.RadioBox(
            self,
            label="Please choose an option:",
            choices=[self.new_project_rbo, self.load_project_rbo],
            majorDimension=0,
            style=wx.RA_SPECIFY_COLS,
        )
        self.sizer.Add(self.proj, pos=(2, 0), span=(1, 2), flag=wx.LEFT, border=5)
        self.proj.Bind(wx.EVT_RADIOBOX, self.chooseOption)

        self.proj_name = wx.StaticText(self, label="Project Name:")
        self.sizer.Add(self.proj_name, pos=(3, 0), span=(1, 2), flag=wx.TOP | wx.LEFT, border=10)

        self.proj_name_txt_box = wx.TextCtrl(self)
        self.sizer.Add(self.proj_name_txt_box, pos=(3, 2), span=(1, 5), flag=wx.TOP | wx.EXPAND, border=10)

    def design_panel_middle(self):
        """
        Creates middle part of the Manage Project page, revised for the use of OpenLabCluster.
        """
        sb = wx.StaticBox(self, label="Optional Attributes")
        self.boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.change_workingdir = wx.CheckBox(
            self, label="Select the Folder Where the Project Will be Created"
        )
        hbox2.Add(self.change_workingdir)
        hbox2.AddSpacer(20)
        self.change_workingdir.Bind(wx.EVT_CHECKBOX, self.activate_change_wd)
        self.sel_wd = wx.Button(self, label="Browse")
        self.sel_wd.Enable(False)
        self.sel_wd.Bind(wx.EVT_BUTTON, self.select_working_dir)
        hbox2.Add(self.sel_wd, 0, wx.ALL, -1)
        self.sel_wd_text = wx.TextCtrl(self, size=(400, 20))
        hbox2.Add(self.sel_wd_text, 0, wx.LEFT, border=10)
        self.boxsizer.Add(hbox2)

        self.sizer.Add(
            self.boxsizer,
            pos=(6, 0),
            span=(1, 10),
            flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=10,
        )

        self.hardware_type = wx.CheckBox(self, label="Use GPU", )
        self.sizer.Add(self.hardware_type, pos=(7, 0), span=(1, 1), flag=wx.TOP | wx.LEFT, border=10)
        self.hardware_type.SetValue(True)

        self.feature_length = wx.StaticText(self, label="Feature Length (int)")
        self.sizer.Add(self.feature_length, pos=(7, 1), span=(1, 1), flag=wx.TOP | wx.LEFT, border=10)
        self.feature_length_txt_box = wx.TextCtrl(self)
        self.sizer.Add(self.feature_length_txt_box, pos=(7, 2), span=(1, 1), flag=wx.TOP | wx.EXPAND, border=10)

        self.cfg_text = wx.StaticText(self, label="Select the Config File")
        self.sizer.Add(self.cfg_text, pos=(8, 0), flag=wx.TOP | wx.EXPAND, border=15)

        if sys.platform == "darwin":
            self.sel_config = wx.FilePickerCtrl(
                self,
                path="",
                style=wx.FLP_USE_TEXTCTRL,
                message="Choose the config.yaml file",
                wildcard="*.yaml",
                size=(300, 50),
            )
        else:
            self.sel_config = wx.FilePickerCtrl(
                self,
                path="",
                style=wx.FLP_USE_TEXTCTRL,
                message="Choose the config.yaml file",
                wildcard="config.yaml",
            )

        self.sizer.Add(self.sel_config, pos=(8, 1), span=(1, 7), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_config.Bind(wx.EVT_BUTTON, self.create_new_project)
        self.sel_config.SetPath("")

        # Hides the button as this is not the default option
        self.sel_config.Hide()
        self.cfg_text.Hide()

    def design_panel_bottom(self):
        """
        Creates bottom part of the Manage Project page, revised for the use of OpenLabCluster.
        """
        self.help_button = wx.Button(self, label="Help")
        self.sizer.Add(self.help_button, pos=(2, 3), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.edit_config_file = wx.Button(self, label="Edit Config File")
        self.sizer.Add(self.edit_config_file, pos=(9, 3))
        self.edit_config_file.Bind(wx.EVT_BUTTON, self.edit_config)
        self.edit_config_file.Enable(False)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(2, 5), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_project)
        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)



    def help_function(self, event):

        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.create_new_project"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")

    def edit_config(self, event):
        """
        """
        if self.cfg != "":
            # For mac compatibility
            if platform.system() == "Darwin":
                self.file_open_bool = subprocess.call(["open", self.cfg])
                self.file_open_bool = True
            else:
                self.file_open_bool = webbrowser.open(self.cfg)

            if self.file_open_bool:
                pass
            else:
                raise FileNotFoundError("File not found!")

    def activate_change_wd(self, event):
        """
        Activates the option to change the working directory.
        """
        self.change_wd = event.GetEventObject()
        if self.change_wd.GetValue() == True:
            self.sel_wd.Enable(True)
        else:
            self.sel_wd.Enable(False)

    def select_working_dir(self, event):
        '''
        Showing a list of available directories
        '''
        cwd = os.getcwd()
        dlg = wx.DirDialog(
            self,
            "Choose the directory where your project will be saved:",
            cwd,
            style=wx.DD_DEFAULT_STYLE,
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.dir = dlg.GetPath()

    def create_new_project(self, event):
        """
        Finally creates the new project.
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
            # Loads previous project
            self.task = self.proj_name_txt_box.GetValue()

    def reset_project(self, event):
        self.loaded = False
        if self.sel_config.IsShown():
            self.sel_config.SetPath("")
            self.proj.SetSelection(0)
            self.sel_config.Hide()
            self.cfg_text.Hide()

        self.sel_config.SetPath("")
        self.proj_name_txt_box.SetValue("")
        self.exp_txt_box.SetValue("")
        self.filelist = []
        self.sel_vids.SetLabel("Load Videos")
        self.dir = os.getcwd()
        self.edit_config_file.Enable(False)
        self.proj_name.Enable(True)
        self.proj_name_txt_box.Enable(True)
        self.multi_choice.Enable(True)
        self.exp.Enable(True)
        self.exp_txt_box.Enable(True)
        self.sel_vids.Enable(True)
        self.addvid.Enable(False)
        self.sel_vids_new.Enable(False)
        self.change_workingdir.Enable(True)
        self.copy_choice.Enable(True)

        try:
            self.change_wd.SetValue(False)
        except:
            pass
        try:
            self.change_copy.SetValue(False)
        except:
            pass
        self.sel_wd.Enable(False)

    def chooseOption(self, event):
        if self.proj.GetStringSelection() == "Load existing project":

            if self.loaded:
                self.sel_config.SetPath(self.cfg)
            self.proj_name.Enable(False)
            self.proj_name_txt_box.Enable(False)
            self.exp.Enable(False)
            self.exp_txt_box.Enable(False)
            self.sel_vids.Enable(False)
            self.sel_vids_new.Enable(False)
            self.change_workingdir.Enable(False)
            self.copy_choice.Enable(False)
            self.multi_choice.Enable(False)
            self.sel_config.Show()
            self.cfg_text.Show()
            self.addvid.Enable(False)
            self.sizer.Fit(self)
        else:
            self.proj_name.Enable(True)
            self.proj_name_txt_box.Enable(True)
            self.exp.Enable(True)
            self.exp_txt_box.Enable(True)
            self.sel_vids.Enable(True)
            self.sel_vids_new.Enable(False)
            self.change_workingdir.Enable(True)
            self.copy_choice.Enable(True)
            self.multi_choice.Enable(True)
            if self.sel_config.IsShown():
                self.sel_config.Hide()
                self.cfg_text.Hide()
            self.addvid.Enable(False)
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
