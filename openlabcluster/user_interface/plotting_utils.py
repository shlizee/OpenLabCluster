"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Developed in UW NeuroAI Lab by Moishe Keselman.
"""
import matplotlib.pyplot as plt


def format_axes(axes: plt.Axes):
    """
    Sets the basic format of plots including font size, font color etc
    """
    font = {
        'family': 'sans-serif',
        'color': 'firebrick',
        'weight': 'normal',
        'size': 16,
    }
    axes_font = {
        'family': 'sans-serif',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
    }

    axes.set_xlabel('Dimension 1', fontdict=axes_font)
    axes.set_ylabel('Dimension 2', fontdict=axes_font)
    if len(axes._get_axis_list()) == 3:
        axes.set_zlabel('Dimension 2', fontdict=axes_font)
    plt.setp(axes.get_yticklabels(), fontweight="normal", size=10)
    plt.setp(axes.get_xticklabels(), fontweight="normal", size=10)
    if len(axes._get_axis_list()) == 3:
        plt.setp(axes.get_zticklabels(), fontweight="normal", size=10)
    axes.set_title(axes.get_title(), fontdict=font)
    axes.set_xlim([-10, 20])
    axes.set_ylim([-10, 20])
    if len(axes._get_axis_list()) == 3:
        axes.set_zlim([-10, 20])
    axes.autoscale()
