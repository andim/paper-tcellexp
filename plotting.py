import string, itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def label_axes(fig_or_axes, labels=string.ascii_uppercase,
               labelstyle=r'{\sf \textbf{%s}}',
               xy=(-0.05, 0.95), xycoords='axes fraction', **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : Where to put the label units (len=2 tuple of floats)
    xycoords : loc relative to axes, figure, etc.
    kwargs : to be passed to annotate
    """
    # re-use labels rather than stop labeling
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    for ax, label in zip(axes, labels):
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords,
                    **kwargs)

class OffsetHandlerTuple(mpl.legend_handler.HandlerTuple):
    """
    Legend Handler for tuple plotting markers on top of each other
    """
    def __init__(self, **kwargs):
        mpl.legend_handler.HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        height *= 1.5
        nhandles = len(orig_handle)
        offset = height / nhandles
        handler_map = legend.get_legend_handler_map()
        a_list = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)
            _a_list = handler.create_artists(legend, handle,
                                             xdescent,
                                             offset*i+ydescent,
                                             width, height,
                                             fontsize,
                                             trans)
            a_list.extend(_a_list)
        return a_list
