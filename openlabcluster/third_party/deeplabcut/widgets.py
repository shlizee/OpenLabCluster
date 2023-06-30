"""
This file is adapted from original file available at
https://github.com/DeepLabCut/DeepLabCut/blob/2472d40a4b1a96130984d9f1bff070f15f5a92a9/deeplabcut/gui/widgets.py
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
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas


class BasePanel(wx.Panel):
    def __init__(self, parent, gui_size=None, projection='2D', **kwargs):
        self.parent = parent
        if gui_size:
            h = gui_size[0]  # / 2
            w = gui_size[1]  # / 3

            wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER, size=(h, w))
        else:
            wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)

        self.sizer = wx.BoxSizer(wx.VERTICAL)


class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        self.panel = wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)
        self.sizer = wx.BoxSizer(wx.VERTICAL)


class ImagePanel(BasePanel):
    """
    Defines an Image Panel for scatter plot visualization and interaction.
    """

    def __init__(self, parent, gui_size=None, projection='2D', **kwargs):
        super(ImagePanel, self).__init__(parent, gui_size=None, projection='2D', **kwargs)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        if projection == '2D':
            self.axes = self.figure.add_subplot(1, 1, 1)
        else:
            self.axes = self.figure.add_subplot(projection='3d')

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.toolbar = NavigationToolbar(self.canvas)

        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)

        self.widgetsizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.zoom = wx.ToggleButton(self, label="Zoom")
        self.zoom.Bind(wx.EVT_TOGGLEBUTTON, self.zoomButton)
        self.widgetsizer.Add(self.zoom, wx.ALL, 15)
        self.zoom.Enable(True)

        self.pan = wx.ToggleButton(self, id=wx.ID_ANY, label="Pan")
        self.widgetsizer.Add(self.pan, wx.ALL, 15)
        self.pan.Bind(wx.EVT_TOGGLEBUTTON, self.panButton)
        self.pan.Enable(True)

        self.orig_xlim = [-60, 60]
        self.orig_ylim = [-60, 60]
        self.sizer.Add(self.widgetsizer)

        self.SetSizer(self.sizer)
        self.resetView()
        self.Fit()

    def renew_sizer(self):
        self.SetSizer(self.sizer)

    def refresh(self, projection):
        wx.Panel.__init__(self, self.parent, -1, style=wx.SUNKEN_BORDER)
        topSplitter = wx.SplitterWindow(self)
        self.axes.remove()
        if projection == '2d':
            self.axes = self.figure.add_subplot(1, 1, 1)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
            self.axes = self.figure.add_subplot(projection='3d')
        self.canvas.draw_idle()

        self.toolbar = NavigationToolbar(self.canvas)
        self.resetView()
        self.Fit()

    def getfigure(self):
        return self.figure

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

    def zoomButton(self, event):
        if self.zoom.GetValue():
            self.pan.SetValue(False)
            # Saves pre-zoom xlim and ylim values
            self.prezoom_xlim = self.axes.get_xlim()
            self.prezoom_ylim = self.axes.get_ylim()
            self.toolbar.zoom()
        else:
            self.toolbar.zoom()

    def panButton(self, event):
        if self.pan.GetValue():
            self.zoom.SetValue(False)
            self.toolbar.pan()
        else:
            self.toolbar.pan()
