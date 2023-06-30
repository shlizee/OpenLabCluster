"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li and Moishe Keselman.
"""
import numpy as np
import wx
import sys
import cv2
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.animation as animation

from wx.core import BoxSizer
from wx.lib.mixins.listctrl import CheckListCtrlMixin, ListCtrlAutoWidthMixin
from wx.lib.scrolledpanel import ScrolledPanel

class CheckListCtrl(wx.ListCtrl, CheckListCtrlMixin, ListCtrlAutoWidthMixin):
    '''
    Creates check list panel
    '''
    def __init__(self, parent, W, H):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT | wx.SUNKEN_BORDER, size=(W, H))
        CheckListCtrlMixin.__init__(self)
        ListCtrlAutoWidthMixin.__init__(self)


class ImagePanel(wx.Panel):
    def __init__(self, parent, gui_size, **kwargs):
        """
        Defines panel to display videos
        """
        # Sets panel with specific size.
        h = gui_size[0] / 2
        w = gui_size[1] / 3
        wx.Panel.__init__(self, parent, -1, style=wx.BORDER_NONE, size=(h, w))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.widgetsizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.orig_xlim = []
        self.orig_ylim = []
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.sizer.Add(self.widgetsizer)
        self.SetSizer(self.sizer)
        self.axes.set_axis_off()
        self.Fit()

    def getfigure(self):
        return self.figure

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

    def renew_sizer(self):
        self.SetSizer(self.sizer)


class Labeling_panel(wx.Panel):
    """
    Defines the sub-window in the Behavior Classification Map panel for displaying and annotating keypoint sequences
    """
    def __init__(self, parent, cfg, current_label, num_labeled_points, update_label, get_selected_id):
        """
        Initializes the video displaying and annotating window
        Inputs:
            parent: parent panel
            cfg: the config file directory
            currently_label: ids of labeled samples
            num_labeled_points: the number of the labeled points
            update_label: function to update the color of a sample once it is labeled
            get_selected_id: function to obtain the ids of selected samples
        """
        # Inits GUI window
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)

        # Gets the number of num_display.
        num_display = (
            wx.Display(i) for i in range(wx.Display.GetCount())
        )
        # Gets the size of each display.
        size_screen = [
            display.GetGeometry().GetSize() for display in num_display
        ]
        # Get screen width and height.
        self.gui_size = (size_screen[0][0] * 0.4, size_screen[0][1] * 0.5)

        # Reads config.
        from openlabcluster.third_party.deeplabcut.auxiliaryfunctions import read_config
        self.cfg_file = cfg
        self.cfg = read_config(cfg)

        # Methods to update plot.
        self.current_label = current_label
        self.update_label = update_label
        self.get_selected_id = get_selected_id
        self.num_labeled_points = num_labeled_points
        label_path = os.path.join(self.cfg['project_path'], self.cfg['label_path'])
        if not os.path.exists(label_path):
            self.total_labelled = 0
        self.load_prelabel()

        class_name = self.cfg['class_name']

        self.SetSizeHints(
            wx.Size(self.gui_size)
        )

        vSplitter = wx.SplitterWindow(self, style=wx.SP_NOBORDER)

        # Defines the image panel and labeling panel.
        self.image_panel = ImagePanel(vSplitter, self.gui_size)
        self.choice_panel = ScrolledPanel(vSplitter)
        vSplitter.SplitVertically(
            self.image_panel, self.choice_panel, sashPosition=self.gui_size[0] * 0.8
        )
        vSplitter.SetSashGravity(1)
        self.widget_panel = self.image_panel.widgetsizer

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(vSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)
        # List of the labeled samples
        self.checked = []
        self.refresh_checkItem = False # True for refresh the panel.
        self.previous_uncheck = False  # Indicator of removing label by unchecking the panel.
        self.log = wx.TextCtrl(self.choice_panel, -1, style=wx.TE_MULTILINE, )
        # Creates list with class names.
        self.list = CheckListCtrl(self.choice_panel, self.gui_size[0] * 0.78, self.gui_size[1] * 0.8)
        self.list.InsertColumn(0, 'Class name', width=180)
        self.list.InsertColumn(1, 'C')
        self.list.OnCheckItem = self.OnCheckItem

        list_sizer = BoxSizer()

        for i in range(len(class_name)):
            index = self.list.InsertItem(sys.maxsize, class_name[i])
            self.list.SetItem(index, 1, str(i + 1))
        list_sizer.Add(self.list)

        self.choice_panel.SetSizer(list_sizer)
        self.choice_panel.SetupScrolling()

        self.pre = wx.Button(self.image_panel, id=wx.ID_ANY, label="   <<Previous<<   ")
        self.widget_panel.Add(self.pre, 1, wx.EXPAND)
        self.pre.Bind(wx.EVT_BUTTON, self.previous)
        self.image_panel.renew_sizer()

        self.prev = wx.Button(self.image_panel, id=wx.ID_ANY, label="Play")
        self.widget_panel.Add(self.prev, 1, wx.EXPAND)
        self.prev.Bind(wx.EVT_BUTTON, self.replay)
        self.prev.Enable(True)
        self.image_panel.renew_sizer()

        self.next_btn = wx.Button(self.image_panel, id=wx.ID_ANY, label=">>Next>>")
        self.widget_panel.Add(self.next_btn, 1, wx.EXPAND)
        self.next_btn.Bind(wx.EVT_BUTTON, self.next)
        self.next_btn.Enable(True)
        self.image_panel.renew_sizer()

    def load_prelabel(self):
        """
        Loads previously annotated labels if possible otherwise initializes labels with all 0
        """
        from openlabcluster.training_utils.ssl.data_loader import get_data_paths
        label_path = os.path.join(self.cfg['project_path'], self.cfg['label_path'])
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        if os.path.exists(os.path.join(label_path, 'label.npy')):
            self.label_full = np.load(os.path.join(label_path, 'label.npy'))
        else:
            import h5py
            train_data = get_data_paths('', self.cfg['data_path'], self.cfg['train'])
            datasize = 0
            for train_data_item in train_data:
                f = h5py.File(train_data_item, 'r')
                if 'label' in list(f.keys()):
                    datasize += len(f['label'])
                else:
                    datasize += 1
            self.label_full = np.zeros(datasize)
        self.total_labelled = np.sum(self.label_full != 0)
        self.total = len(self.label_full)

    def OnCheckItem(self, index, flage):
        """
        Defines a window allowing users to annotate the behavior classes of sequences based on the corresponding videos
        """
        print(self.refresh_checkItem)
        if self.refresh_checkItem:
            self.refresh_checkItem = False
            self.checked = []
        else:
            if len(self.checked):
                # With previous checked class.
                if self.checked[0] == index:
                    # Unchecks the class
                    self.checked = []

                    if self.label_full[self.sample[self.current]] != 0:
                        if not self.previous_uncheck:
                            # Changes the label of previously labeled sample.
                            self.total_labelled -= 1
                            self.label_full[self.sample[self.current]] = 0
                            self.update_label(self.sample[self.current], uncheck=True)
                            self.num_labeled_points()
                        else:
                            self.previous_uncheck = False

                else:
                    # Changes the checked class.
                    self.list.CheckItem(self.checked[0], False)
                    print('before check', self.checked)
                    self.checked = [index]
                    print('after checked', self.checked)
                    if self.label_full[self.sample[self.current]] == 0:
                        self.total_labelled += 1
                    self.label_full[self.sample[self.current]] = index + 1
                    self.update_label(self.sample[self.current])
                    self.num_labeled_points()

            else:

                self.checked = [index]

                if self.label_full[self.sample[self.current]] == 0:
                    self.total_labelled += 1
                self.label_full[self.sample[self.current]] = index + 1
                print('video_name:%s' % self.videpaths[self.sample[self.current]])
                self.update_label(self.sample[self.current])
                # Updates the number of labeled points.
                self.num_labeled_points()

    def animate(self, i):
        """
        Video animation
        """
        self.videohandle.set_array(self.imgs[i])
        return self.videohandle,

    def load_video(self):
        """
        Displays the video of the current sample
        """
        if len(self.checked) > 0:
            self.refresh_checkItem = True
            self.list.CheckItem(self.checked[0], False)
        self.current = 0
        self.sample = self.get_selected_id()
        self.current_label(self.sample[self.current])
        self.num_labeled_points()

        # Get all images for a video and plays them
        if self.cfg['train_videolist'].endswith('.text') or self.cfg['train_videolist'].endswith('.txt'):
            with open(self.cfg['train_videolist'], 'r') as f:
                self.videpaths = f.readlines()
            # Selection start from self.current so set 0 here
            print(self.cfg['train_videolist'])
            print(self.sample[self.current])
            path = self.videpaths[self.sample[self.current]][:-1]
            cap = cv2.VideoCapture(path)
            self.imgs = []
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    self.imgs.append(frame)
                else:
                    break
        elif self.cfg['train_videolist'].endswith('.npy'):
            self.videpaths = np.load(self.cfg['train_videolist'], allow_pickle=True)
            path = self.videpaths[self.sample[self.current]]
            self.imgs = []
            for i in range(len(path)):
                self.imgs.append(cv2.imread(path[i]))

        if len(self.imgs) > 0:
            self.videohandle = self.image_panel.axes.imshow(self.imgs[0])

        animation.FuncAnimation(self.image_panel.figure, self.animate, frames=len(self.imgs),
                                interval=1, blit=True, repeat=False)
        self.image_panel.canvas.draw_idle()

    def next(self, event):
        """
        Goes to the next sample in the annotation list
        """
        self.sample = self.get_selected_id()
        print('current from next', self.current)

        # Disables next if there is no checked item from previous
        if not len(self.checked):
            return
        if self.current + 1 < len(self.sample):
            self.current += 1

            if self.label_full[self.sample[self.current]] != 0:
                # Keeps earlier labeled results
                if len(self.checked):
                    self.previous_uncheck = True
                    self.list.CheckItem(self.checked[0], False)
                self.list.CheckItem(self.label_full[self.sample[self.current]] - 1, True)
            else:
                if len(self.checked):
                    self.list.CheckItem(self.checked[0], False)

            if self.label_full[self.sample[self.current - 1]] != 0:
                self.update_label(self.sample[self.current - 1])
            self.current_label(self.sample[self.current])
            print('finished current label')
            if self.cfg['train_videolist'].endswith('.text') or self.cfg['train_videolist'].endswith('.txt'):
                path = self.videpaths[self.sample[self.current]][:-1]
                cap = cv2.VideoCapture(path)
                self.imgs = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        self.imgs.append(frame)
                    else:
                        break
            elif self.cfg['train_videolist'].endswith('.npy'):
                self.videpaths = np.load(self.cfg['train_videolist'], allow_pickle=True)
                path = self.videpaths[self.sample[self.current]]
                self.imgs = []
                for i in range(len(path)):
                    self.imgs.append(cv2.imread(path[i]))

            self.videohandle = self.image_panel.axes.imshow(self.imgs[0])
            print(len(self.imgs))
            animation.FuncAnimation(self.image_panel.figure, self.animate, frames=len(self.imgs),
                                    interval=1, blit=True, repeat=False)
        else:
            wx.MessageBox(
                "This is the End of Sampling list",
                "Info",
                wx.OK | wx.ICON_ERROR,
            )
            return

    def previous(self, event):
        """
        Goes to the previous sample for in the annotation list
        """
        self.sample = self.get_selected_id()
        if self.current - 1 >= 0:
            self.current -= 1

            if len(self.checked):
                self.previous_uncheck = True
                self.list.CheckItem(self.checked[0], False)
            if self.label_full[self.sample[self.current]] != 0:
                self.list.CheckItem(self.label_full[self.sample[self.current]] - 1, True)
            self.current_label(self.sample[self.current], self.sample[self.current + 1])
            if self.cfg['train_videolist'].endswith('.text') or self.cfg['train_videolist'].endswith('.txt'):
                path = self.videpaths[self.sample[self.current]][:-1]
                cap = cv2.VideoCapture(path
                                       )
                self.imgs = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        self.imgs.append(frame)
                    else:
                        break
                cap.release()
            elif self.cfg['train_videolist'].endswith('.npy'):
                self.videpaths = np.load(self.cfg['train_videolist'], allow_pickle=True)
                path = self.videpaths[self.sample[self.current]]
                self.imgs = []
                for i in range(len(path)):
                    self.imgs.append(cv2.imread(path[i]))

            self.videohandle = self.image_panel.axes.imshow(self.imgs[0])

            animation.FuncAnimation(self.image_panel.figure, self.animate, frames=len(self.imgs),
                                    interval=1, blit=True, repeat=False)
        else:
            wx.MessageBox(
                "This is the End of Sampling list",
                "Info",
                wx.OK | wx.ICON_ERROR,
            )
            return

    def replay(self, event):
        """
        Replays the video of the current sample
        """
        if len(self.imgs) == 0:
            wx.MessageDialog(self, 'Load video first', 'Error', style=wx.OK | wx.CENTRE)
        else:
            animation.FuncAnimation(self.image_panel.figure, self.animate, frames=len(self.imgs),
                                    interval=1, blit=True, repeat=False)

    def savelabel(self):
        """
        Saves annotations into a .npy file
        """
        label_path = os.path.join(self.cfg['project_path'], self.cfg['label_path'])
        np.save(os.path.join(label_path, 'label.npy'), self.label_full)


class video_display_window(wx.Panel):
    """Video displaying window for the Cluster Map page."""

    def __init__(self, parent, gui_size, cfg, **kwargs):
        """
        Initializes a video window in the Cluster Map.
        """
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.orig_xlim = []
        self.orig_ylim = []
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.axes.set_axis_off()
        self.Fit()
        self.cfg = cfg

    def animate(self, i, vid_id):
        """
        Video animation
        """
        if vid_id == self.animation_vid_id:
            self.videohandle.set_array(self.imgs[i])
        return self.videohandle,

    def load_video(self, sequence_id):
        """
        Displays the video of a keypoint sequence chosen by users in the clustering plot
        """
        fps = 10
        if type(self.cfg['train_videolist']) is str:
            if self.cfg['train_videolist'].endswith('.text') or self.cfg['train_videolist'].endswith('.txt'):
                with open(self.cfg['train_videolist'], 'r') as f:
                    self.videpaths = f.readlines()
                # Selection start from self.current so set 0 here
                path = self.videpaths[sequence_id].replace('\n', '').encode('ascii', 'ignore').decode('utf-8')
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.imgs = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        self.imgs.append(frame)
                    else:
                        break
            elif self.cfg['train_videolist'].endswith('.npy'):
                self.videpaths = np.load(self.cfg['train_videolist'], allow_pickle=True)
                paths = self.videpaths[sequence_id]
                self.imgs = []
                for i in range(len(paths)):
                    self.imgs.append(cv2.imread(paths[i]))

        try:
            self.videohandle = self.axes.imshow(self.imgs[0])
        except:
            wx.MessageBox("Failed to load video", 'Error', wx.OK | wx.ICON_ERROR)

        print(len(self.imgs))
        self.animation_vid_id = np.random.randint(0, 1000000)
        self.animation = animation.FuncAnimation(self.figure, self.animate, frames=len(self.imgs),
                                                 interval=1000 / fps, blit=True, repeat=False,
                                                 fargs=(self.animation_vid_id,))
        self.canvas.draw_idle()


def show(cfg, sample):
    app = wx.App(redirect=0)
    GUI = Labeling_panel(None, cfg, sample)
    GUI.Show()
    app.MainLoop()
