"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/d905e14b2343667e38b8477f28841671e615abce/deeplabcut/gui/welcome.py
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
import wx.adv
from wx.lib.scrolledpanel import ScrolledPanel

def scale_bitmap(bitmap, width, height):
    image = wx.ImageFromBitmap(bitmap)
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.BitmapFromImage(image)
    return result

class Welcome(wx.lib.scrolledpanel.ScrolledPanel):
    """
    Defines the Welcome panel in OpenLabCluster based on the Welcome panel of DeepLabCut.  
    """
    def __init__(self, parent, gui_size, image_path):
        h = gui_size[0]
        w = gui_size[1]
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent=parent)
        sizer = wx.BoxSizer(wx.VERTICAL)

        icon = wx.StaticBitmap(self, -1, bitmap=wx.Bitmap(image_path))
        sizer.Add(icon, flag=wx.ALIGN_CENTRE | wx.ALL, border=10)
        line = wx.StaticLine(self)
        sizer.Add(line,  flag=wx.EXPAND | wx.BOTTOM, border=10)

        # If editing this text make sure you add the '\n' to get the new line. The sizer is unable to format lines correctly.
        description = "OpenLabCluster is an open source tool for \n" \
                      "clustering and classification of behavior video.\n" \
                      "To start, please select the \"Manage Project\" Tab."

        self.proj_name = wx.StaticText(self, label=description, style=wx.ALIGN_CENTRE)
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        self.proj_name.SetFont(font)
        sizer.Add(self.proj_name, flag=wx.ALIGN_CENTRE, border=0)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.SetupScrolling()
