"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li and Moishe Keselman.
"""
import os
from os.path import dirname as dirname
import wx
import sys

# Third party import
import openlabcluster
from openlabcluster.third_party.deeplabcut.launch_script import MainFrame
from openlabcluster.third_party.deeplabcut.welcome import Welcome

# OpenLabCluster import
from openlabcluster.user_interface.init_proj import Project_constructor

print(sys.path)
sys.path.insert(1, dirname(dirname(dirname(os.path.realpath(__file__)))))


class StartGUI(MainFrame):
    """
    This is the entrance to the OpenLabCluster GUI
    Notes:
    The class inherits class MainFrame from third_party
    """

    def __init__(self):
        """
        Initializes the OpenLabCluster GUI.
        """
        super(StartGUI, self).__init__()

        # Initializes statusbar to show status of points.
        self.statusBar = wx.StatusBar(self, -1)
        self.SetStatusBar(self.statusBar)
        self.statusBar.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL))
        self.statusBar.SetMinHeight(30)
        self.statusBar.SetBackgroundColour((175, 255, 212, 156))

        # Sets font size globally.
        self.SetFont(wx.Font(15, wx.SWISS, wx.NORMAL, wx.NORMAL, False, 'MS Shell Dlg 2'))

        self.SetSizeHints(
            wx.Size(self.gui_size)
        )
        self.create_panel(Project_constructor, self.statusBar)

    def create_panel(self, project_module, status_bar=None):
        """
        Initializes Welcome page and Manage Project page
        """
        # Creates a notebook.
        self.panel = wx.Panel(self)
        self.nb = wx.Notebook(self.panel, style=wx.NB_TOP)

        # Adds Welcome and Manage Project windows as pages of the notebook.
        media_path = os.path.join(openlabcluster.__path__[0], "user_interface", "images")
        image_path = os.path.join(media_path, "GUIplot.png")
        page1 = Welcome(self.nb, self.gui_size, image_path)
        self.nb.AddPage(page1, "Welcome")

        page2 = project_module(self.nb, self.gui_size, status_bar)
        self.nb.AddPage(page2, "Manage Project")

        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.panel.SetSizer(self.sizer)

# launches OpenLabCluster.
def launch_olc():
    app = wx.App()
    frame = StartGUI().Show()
    app.MainLoop()


if __name__ == '__main__':
    app = wx.App()
    frame = StartGUI().Show()
    app.MainLoop()
