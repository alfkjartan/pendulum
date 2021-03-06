""" Defines a set of matplotlib figures which will show a live view of a set of time series. """

# Kjartan Halvorsen
# 2013-05-20

import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesPlot:

    def __init__(self, fig, rect):
        self.fig = fig # The figure to add plots to
        self.animated = []
        self.rect = rect
        
    def init_plot(self, ts, mintime=10):
        """ Initializes the plot. ts must be a list of dicts describing the plots.
        The dict contains the field name and timeseries, where timeseries is a list
        of data to plot. Its items are kwargs that can be sent to axes.plot.
        """
        self.plot_data = {}
        self.time_data = []
        
        self.mintime = mintime

        plts = []
        lbls = []

        nplots = len(ts)
        k = 0
        margin = 0.05
        bottom = margin
        height = (self.rect[3]-(nplots-1)*margin) / nplots 
        for subplot in ts:
            #ax = self.fig.add_subplot(nplots, 1, k)
            
            ax = self.fig.add_axes([self.rect[0], bottom, self.rect[2], height])
            ax.set_axis_bgcolor('black')
            ax.grid(True, color='gray')
            ax.set_title(subplot['name'], fontsize=8)
            ax.set_ylim(subplot['ylim'])
            
            plt.setp(ax.get_xticklabels(), fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)

            extremes = np.array([1e12, -1e12])

            bottom += height + margin
            
            for p in subplot['timeseries']:
                lbl = p['label']
                pl = ax.plot(self.time_data, [], **p)
                lbls.append(lbl)
                plts.append(pl[0])
                #self.plot_data[lbl] = ( [], pl[0], ax, extremes, canvas, canvas.copy_from_bbox(ax.bbox) )
                self.plot_data[lbl] = ( [], pl[0], ax, extremes)
                self.animated.append(pl[0])
                
            ax.set_ylim(subplot['ylim'])
        self.fig.legend(plts, lbls, 'upper right')

        plt.draw()
    def append_data(self, t, dta):
        self.time_data.append(t)
        for (label, dta_) in dta.iteritems():
            #(linedata, lineplot, ax, minmax, canvas, background) =  self.plot_data[label]
            (linedata, lineplot, ax, minmax) =  self.plot_data[label]

            # Restore clean slate background
            #canvas.restore_region(background)

            linedata.append(dta_)
            lineplot.set_data(self.time_data, linedata)

            # just draw the animated artist
            #ax.draw_artist(lineplot)
            # just redraw the axes rectangle
            #canvas.blit(ax.bbox)

            ax.set_xbound(lower=0, upper=max(self.mintime, t))
            # Update min and max values
            #if dta > minmax[1] :
            #    minmax[1] = dta_
            #    ax.set_ybound(lower=minmax[0], upper=minmax[1])
            #    if dta < minmax[0]:
            #        minmax[0] = dta_
            #        ax.set_ybound(lower=minmax[0], upper=minmax[1])

            #self.fig.canvas.draw()
           
