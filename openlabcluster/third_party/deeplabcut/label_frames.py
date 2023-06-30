"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/gui/label_frames.py
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

class Label_frames(wx.lib.scrolledpanel.ScrolledPanel):
    """
    """
    def __init__(self, parent, gui_size, cfg):
        """Constructor"""
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent=parent)
        self.config = cfg

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.parent = parent
        title = wx.StaticText(self, label="OpenLabCluster - Step 3. Behavior Classification Map with Active Learning")
        self.sizer.Add(title, 0, flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.BOTTOM, border=15)
        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, 0, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=5)

    def help_function(self, event):
        filepath = "help.txt"
        f = open(filepath, "w")
        sys.stdout = f
        fnc_name = "deeplabcut.label_frames"
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt", "r+")
        help_text = help_file.read()
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove("help.txt")
