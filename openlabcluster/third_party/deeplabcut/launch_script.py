"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/gui/launch_script.py
"""

"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import wx

class MainFrame(wx.Frame):
    def __init__(self):
        """
        Initializes GUI window revised for the use of OpenLabCluster.
        """
        displays = (
            wx.Display(i) for i in range(wx.Display.GetCount())
        )  # Gets the number of displays
        screenSizes = [
            display.GetGeometry().GetSize() for display in displays
        ]  # Gets the size of each display
        index = 0  # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth * 0.9, screenHeight * 0.85)

        wx.Frame.__init__(
            self,
            None,
            wx.ID_ANY,
            "OpenLabCluster",
            size=wx.Size(self.gui_size),
            pos=wx.DefaultPosition,
            style=wx.RESIZE_BORDER | wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )


