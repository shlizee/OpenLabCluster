"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import numpy as np
import wx

# Third-party import
from openlabcluster.third_party.deeplabcut.auxiliaryfunctions import read_config

# OpenLabCluster import
from openlabcluster.user_interface.plotting_utils import format_axes
from openlabcluster.user_interface.cluster_map import ImagePanel
from openlabcluster.user_interface.plot_hidden import extract_hid


class PlotGUI_panel(wx.Panel):
    """
    Defines the interactive sample selection window in the Behavior Classification Map.
    """

    def __init__(self, parent, cfg, sample_method, num_samples, reducer_name='PCA', dimension='2d', model_name=None,
                 model_type=None):
        """Defines basic functions of the sample selection window.
        Inputs:
            parent: parent window handle.
            cfg: the config file directory.
            sample_method: the name of active learning method for label selection
                           (options: "Marginal Index (MI)", "Core Set (CS)", "Cluster Center (Top)",
                            "Cluster Random (Rand)", "Uniform (Uni)").
            num_samples: the number of samples to select in each iteration.
            reducer_name: the name of dimension reduction methods,(options: "PCA", "tSNE", "UMAP").
            dimension: the number of dimensions to for behavior classification map plot, either "2d" or "3d".
            model_name: the directory of the pretrained model.
            model_type: the type of model, either "seq2seq" or "semi_seq2seq".
        """
        self.cfg = cfg
        self.cfg_data = read_config(self.cfg)

        self.reducer_name = reducer_name
        self.dimension = dimension
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.cfg_data['tr_modelName']
            if not self.model_name:
                wx.MessageBox('No Model Exist Yet!', 'Error')
                return

        # Inits GUI window
        wx.Panel.__init__(self, parent, -1, style=wx.BORDER_NONE)
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
        self.selection_id = {"Uniform (Uni)": 'random', "Cluster Center (Top)": 'ktop',
                             "Cluster Random (Rand)": 'krandom', "Marginal Index (MI)": 'kmi',
                             "Core Set (CS)": 'core_set'}

        self.SetSizeHints(
            wx.Size(self.gui_size)
        )

        # Defines sub-panels showing plots.
        self.image_panel = ImagePanel(self, self.gui_size)
        self.image_panel.axes.set_title('Behavior Classification Map')

        format_axes(self.image_panel.axes)

        self.num_sample = num_samples
        self.image_panel.renew_sizer()
        self.sizer = wx.GridBagSizer(1, 1)
        self.sizer.Add(self.image_panel, pos=(0, 0), span=(1, 1))
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        # Sets default parameters.
        if model_type:
            self.model_type = model_type
        else:
            self.model_type = self.cfg_data['tr_modelType']

        if sample_method:
            self.sample_method = self.selection_id[sample_method]
        else:
            self.sample_method = self.selection_id.get(self.cfg_data['sample_method'], 'ktop')
        self.selected_id = []
        self.labeled_id = []
        self.current = []
        self.text = ['Selected Points\n']
        if self.model_name:
            # Computes hidden state of each sample and gets the id of labeled samples.
            toLabel = self.extract_hiddenstate()
            self.labeled_id += toLabel
            # Initial scatter plot.
            self.plot_initial()
            # Annotates samples that have been labeled.
            self.plot_iter()
            # Annotates samples that are suggested for labeling.
            self.plot_suggest()
            self.highlight, = self.image_panel.axes.plot([], [], 'o', color='r')
            self.display_status = True
            self.image_panel.canvas.mpl_connect('pick_event', self.display_data)

        self.image_panel.axes.autoscale()

    def refresh(self, event, selection_method):
        """
        Re-plots the selected sample with the updated model or updated selection method.
        """
        self.selected_id = []
        from openlabcluster.third_party.deeplabcut.auxiliaryfunctions import read_config
        self.image_panel.axes.clear()
        format_axes(self.image_panel.axes)
        self.image_panel.axes.set_title('Behavior Classification Map')
        self.cfg_data = read_config(self.cfg)

        self.model_name = self.cfg_data['tr_modelName']
        if not self.model_name:
            wx.MessageBox('No Model Exist Yet!', 'Error')
            return

        self.model_type = self.cfg_data['tr_modelType']

        print('selection methods', self.selection_id[selection_method])

        self.sample_method = self.selection_id[selection_method]

        # Get suggested label using hidden state.
        self.extract_hiddenstate()
        self.plot_initial()
        self.plot_iter()
        self.plot_suggest()
        self.load_video()

    def savelabel(self, event):
        """
        Saves the annotation results.
        """
        self.load_video()

    def initialize_video_connection(self, load_video):
        """
        Connects the samples selection window with the video annotation window.
        """
        self.load_video = load_video

    def active_display(self):
        """
        Enables re-selection of samples for annotation.
        """
        self.display_status = True

    def deactive_display(self):
        """
        Disables re-selection of samples for annotation.
        """
        self.display_status = False

    def display_data(self, event):
        """
        Displays hidden states in the reduced dimension space.
        """
        if self.display_status == True:
            cur_pos = np.array([event.mouseevent.xdata, event.mouseevent.ydata])
            inarray = np.asarray(event.ind)
            print(inarray)
            if len(inarray) > 0:
                x_tmp = self.extracted.transformed[inarray, 0]
                y_tmp = self.extracted.transformed[inarray, 1]
                if self.dimension == '3d':
                    z_tmp = self.extracted.transformed[inarray, 2]
                    dist = (x_tmp - cur_pos[0]) ** 2 + (y_tmp - cur_pos[1]) ** 2
                else:
                    dist = (x_tmp - cur_pos[0]) ** 2 + (y_tmp - cur_pos[1]) ** 2
                ind = inarray[np.argmin(dist)]
            else:
                ind = inarray

            if isinstance(ind, list):
                ind = ind[0]
            if ind not in self.labeled_id and ind not in self.current:
                if ind not in self.selected_id:
                    self.selected_id = self.selected_id + [ind]
                    print('id', self.selected_id)
                    if self.dimension == '2d':
                        self.image_panel.axes.plot(self.extracted.transformed[ind, 0],
                                                   self.extracted.transformed[ind, 1], 'o', color='blue')

                    else:
                        self.image_panel.axes.plot3D(self.extracted.transformed[ind, 0],
                                                     self.extracted.transformed[ind, 1],
                                                     self.extracted.transformed[ind, 3], 'o',
                                                     color='blue')
                    self.text = self.text + ['P:%d\n' % ind]
                    self.image_panel.canvas.draw_idle()
                else:
                    self.selected_id.remove(ind)
                    self.text.remove('P:%d\n' % ind)
                    if self.dimension == '2d':
                        self.image_panel.axes.plot(self.extracted.transformed[ind, 0],
                                                   self.extracted.transformed[ind, 1], 'o',
                                                   color='grey')

                    else:
                        self.image_panel.axes.plot3D(self.extracted.transformed[ind, 0],
                                                     self.extracted.transformed[ind, 1],
                                                     self.extracted.transformed[ind, 3], 'o',
                                                     color='grey')
                    self.image_panel.canvas.draw_idle()

    def extract_hiddenstate(self):
        """
        Extracts hidden states with the trained encoder model.
        """
        import os
        import re
        self.cfg_data = read_config(self.cfg)
        label_path = os.path.join(self.cfg_data['project_path'], self.cfg_data['sample_path'])
        self.model_type = self.cfg_data['tr_modelType']
        self.model_name = self.cfg_data['tr_modelName']
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        file_list = [x for x in os.listdir(label_path) if not x.startswith('.')]
        print(file_list)
        if len(file_list) > 0:
            last = np.sort(file_list)[-1]
            iter = re.split('([0-9]+)', last)
            self.iter = int(iter[1]) + 1
        else:
            self.iter = 0

        self.extracted = extract_hid(self.cfg, dimension=self.dimension, reducer_name=self.reducer_name)

        # Imports functions for active learning.
        from openlabcluster.training_utils.ssl.clustering_classification import iter_kmeans_cluster, \
            remove_labeled_cluster
        from openlabcluster.training_utils.ssl.labelSampling import SampleFromCluster
        from openlabcluster.third_party.kcenter.kcenter_greedy import kCenterGreedy

        # Gets already labeled samples and suggests potential samples for labeling.
        toLabel = np.where(self.extracted.semilabel != 0)[0].tolist()
        index_train_complete = np.arange(0, self.extracted.hidarray.shape[0])
        if self.model_type == 'seq2seq':

            if self.sample_method == 'core_set':
                self.cor_set = kCenterGreedy(self.extracted.hidarray, None, seed=1)
                self.suggest = self.cor_set.select_batch_(None, toLabel, self.num_sample)
            else:
                hi_train, index_train = remove_labeled_cluster(self.extracted.hidarray,
                                                               index_train_complete, toLabel)
                train_id_list, dis_list, dis_list_prob, label_list = iter_kmeans_cluster(hi_train, index_train,
                                                                                         self.num_sample)
                self.suggest = SampleFromCluster(train_id_list, dis_list, dis_list_prob, 'ktop',
                                                 self.num_sample)
        else:
            if self.sample_method == 'core_set':
                self.cor_set = kCenterGreedy(self.extracted.hidarray, None, seed=1)
                self.suggest = self.cor_set.select_batch_(None, toLabel, self.num_sample)
            else:
                hi_train, index_train, mi, = remove_labeled_cluster(self.extracted.hidarray,
                                                                    index_train_complete, toLabel, self.extracted.mi)
                try:
                    test = mi[0]
                except:
                    wx.MessageBox('All samples have been labled!', 'Error')
                train_id_list, dis_list, dis_list_prob, label_list = iter_kmeans_cluster(hi_train, index_train,
                                                                                         self.num_sample, mi)
                print('sample from cluster', self.sample_method)
                self.suggest = SampleFromCluster(train_id_list, dis_list, dis_list_prob, self.sample_method,
                                                 self.num_sample)
        return toLabel

    def plot_initial(self):
        """
        Initializes the scatter plot in the sample selection window.
        """
        self.image_panel.axes.scatter(self.extracted.transformed[:, 0], self.extracted.transformed[:, 1], picker=True,
                                      color='grey')

    def plot_suggest(self):
        """
        Highlights the samples that are selected and waiting for labeling in blue.
        """
        for ind in self.suggest:
            self.selected_id = self.selected_id + [ind]
            print('id', self.selected_id)
            self.image_panel.axes.scatter(self.extracted.transformed[ind, 0], self.extracted.transformed[ind, 1],
                                          color='blue')

            self.text = self.text + ['P:%d\n' % ind]
        self.image_panel.canvas.draw_idle()

    def plot_iter(self):
        """
        Changes the color of samples that have been annotated to green.
        """
        i = self.extracted.semilabel != 0
        if sum(i) != 0:
            self.image_panel.axes.plot(self.extracted.transformed[i, 0], self.extracted.transformed[i, 1], 'o',
                                       color='green', alpha=1)

    def current_sample(self, index, index_previous=None):
        """
        Changes the color of the current sample waiting for annotation to red.
        """
        self.current = [index]
        self.image_panel.axes.plot(self.extracted.transformed[index, 0], self.extracted.transformed[index, 1], 'o',
                                   color='r')
        print('draw index', index)
        self.image_panel.canvas.draw_idle()
        if index_previous != None:
            if index_previous not in self.labeled_id:
                self.image_panel.axes.plot(self.extracted.transformed[index_previous, 0],
                                           self.extracted.transformed[index_previous, 1], 'o',
                                           color='b')
                self.image_panel.canvas.draw_idle()
            else:
                self.image_panel.axes.plot(self.extracted.transformed[index_previous, 0],
                                           self.extracted.transformed[index_previous, 1], 'o',
                                           color='g')
                self.image_panel.canvas.draw_idle()

    def update_labeled(self, index, uncheck=False):
        """
        Changes the color of the recent labeled sample to green.
        """
        if not uncheck:
            self.image_panel.axes.plot(self.extracted.transformed[index, 0], self.extracted.transformed[index, 1], 'o',
                                       color='g')
            if index not in self.labeled_id:
                self.labeled_id.append(index)
        else:
            self.image_panel.axes.plot(self.extracted.transformed[index, 0], self.extracted.transformed[index, 1], 'o',
                                       color='r')
            if index in self.labeled_id:
                self.labeled_id.remove(index)
        self.image_panel.canvas.draw_idle()

    def get_suggest(self):
        """
        Gets the ids of the selected samples.
        """
        return self.selected_id

    def update_image_panel(self, dim, method):
        """
        Updates the image panel whenever there is an update in the trained model, the dimension reduction method, or the number of dimensions.
        """
        self.dimension = dim
        self.reducer_name = method
        self.image_panel.refresh(dim)
        self.extracted = extract_hid(self.cfg, dimension=self.dimension, reducer_name=self.reducer_name,
                                    )
        transform = self.extracted.transformed
        if dim == '2d':
            self.sc = self.image_panel.axes.scatter(transform[:, 0], transform[:, 1], s=10, picker=True, color='k')
        else:
            self.sc = self.image_panel.axes.scatter(transform[:, 0], transform[:, 1], transform[:, 2], s=10,
                                                    picker=True, color='k')
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }
        self.image_panel.axes.set_title('Behavior Classification Map', fontdict=font)
        self.image_panel.axes.autoscale()
        format_axes(self.image_panel.axes)


def show(cfg, model_name, model_type, sample_method, percentage):
    app = wx.App(redirect=0)
    GUI = PlotGUI_panel(None, cfg, model_name, model_type, sample_method, percentage)
    GUI.Show()
    app.MainLoop()
