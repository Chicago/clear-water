import read_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import sklearn.metrics

# Should we block the matplotlib plots?
TO_BLOCK = True


def roc(scores, labels, block_show=TO_BLOCK, ax=None, bounds=None, mark_threshes=None):
    '''
    Plots the Receiver Operator Characteristic (ROC) curve of the results
    from a binary classification system.

    Inputs
    ------
    scores : Nx1 array of (usually continuous) outputs of a classification
             system, for example, predicted E. coli levels
    labels : Nx1 boolean array of true classes

    Keyword arguments
    -----------------
    block_show : Boolean, if true, then the call to matplotlib.pyplot.show()
                 will block further computation until the window is closed.
    ax         : Will be plotted to the specified axis, or a new axis
                 if ax is None.
    bounds     : The x limits (FPR limits) of the ROC curve are restricted
                 to these specified bounds. If None, then [0, 1] are used.
    mark_threshes :
                 A list of thresholds to mark with an X.

    Returns
    -------
    fpr      : Mx1 array of false positive rates
    tpr      : Mx1 array of true positive rates
    threshes : Mx1 array of the thresholds on the scores variable
               used to create the corresponding fpr and tpr values.
    bounds   : [min, max] bounds of the ROC curve, used for setting
               axis limits as well as computing the AUC. If None, the
               bounds [0, 1] are used.

    Example
    -------
    >>> import read_data as rd
    >>> import visualizations as viz
    >>> df = rd.read_data()
    >>> scores = df[['Reading.1', 'Escherichia.coli']].dropna()['Reading.1']
    >>> labels = df[['Reading.1', 'Escherichia.coli']].dropna()['Escherichia.coli']
    >>> labels = labels >= 235.0
    >>> viz.roc(scores, labels)
    >>> # Warning! This will perform very well b/c it's predcting today
    >>> # using today's data, this is not a viable model!
    '''

    fpr, tpr, threshes = sklearn.metrics.roc_curve(labels, scores)

    ax.plot(fpr, tpr)
    if mark_threshes:
        idxs = [np.where(threshes < t)[0][0] for t in mark_threshes]
        ax.plot(fpr[idxs], tpr[idxs], 'xk', markersize=10.0, markeredgewidth=2.0)
    ax.hold(True)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')

    if bounds:
        ax.set_xlim(bounds)
        flt = (fpr >= bounds[0]) & (fpr <= bounds[1])
        ax.set_ylim([0, min(1, max(tpr[flt]) * 1.1)])
        auc = np.trapz(tpr[flt], x=fpr[flt])
    else:
        ax.set_aspect('equal')
        auc = np.trapz(tpr, x=fpr)

    plt.draw()
    plt.show(block=block_show)


    return fpr, tpr, threshes, auc


def precision_recall(scores, labels, block_show=TO_BLOCK, ax=None):
    '''
    Plots the Precision Recall (PR) curve of the results
    from a binary classification system.

    Inputs
    ------
    scores : Nx1 array of (usually continuous) outputs of a classification
             system, for example, predicted E. coli levels
    labels : Nx1 boolean array of true classes

    Keyword arguments
    -----------------
    block_show : Boolean, if true, then the call to matplotlib.pyplot.show()
                 will block further computation until the window is closed.
    ax         : Will be plotted to the specified axis, or a new axis
                 if ax is None.

    Returns
    -------
    tpr      : Mx1 array of true positive rates
    ppv      : Mx1 array of positive predictive values
    threshes : Mx1 array of the thresholds on the scores variable
               used to create the corresponding ppv and tpr values.

    Example
    -------
    >>> import read_data as rd
    >>> import visualizations as viz
    >>> df = rd.read_data()
    >>> scores = df[['Reading.1', 'Escherichia.coli']].dropna()['Reading.1']
    >>> labels = df[['Reading.1', 'Escherichia.coli']].dropna()['Escherichia.coli']
    >>> labels = labels >= 235.0
    >>> viz.precision_recall(scores, labels)
    >>> # Warning! This will perform very well b/c it's predcting today
    >>> # using today's data, this is not a viable model!
    '''

    ppv, tpr, threshes = sklearn.metrics.precision_recall_curve(labels, scores)

    if ax is None:
        ax = plt.subplots(1)[1]

    ax.plot(tpr, ppv)
    ax.hold(True)
    ax.plot([0, 1], [float(labels.sum()) / labels.size,
                     float(labels.sum()) / labels.size], 'r--')
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('Positive Predictive Value')
    ax.set_title('PR curve')
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.show(block=block_show)

    auc = np.trapz(ppv, x=tpr[::-1])

    return tpr, ppv, threshes, auc


def beach_hist(col='Escherichia.coli', beaches=None,
               subplots=False, transform=lambda x: x, df=None):
    '''
    Plots histograms of a specified column for the specified beaches.

    Inputs
    ------
    col       : Column name or index of the column to be histogrammed
    beaches   : List of beach names to generate histograms for, None indicates
                that all beaches should be used.
    subplots  : False to have each beach's histogram be plotted on the same
                axis. Otherwise, subplots is a list with two elements specifying
                the dimensions of the subplot array. For example, [8, 4] will
                create an 8x4 grid of subplots. There must be at least as many
                subplot axes as beaches.
    transform : A function to trasform the data, can be useful to log scale
                the E. coli readings to make the histogram more spread out.
    df        : The dataframe containing the data. If None, the data will be
                read in using read_data.

    Example
    -------
    >>> import read_data as rd
    >>> import visualizations as viz
    >>> import numpy as np
    >>> df = rd.read_data()
    >>> # Will be very messy, you should only plot on the same axis when there
    >>> # are only a few beaches to plot
    >>> viz.beach_hist(transform=lambda x: np.log(x+1), df=df)
    >>> viz.beach_hist(transform=lambda x: np.log(x+1), df=df, subplots=[7, 4])
    '''

    if df is None:
        df = read_data.read_data()

    if beaches is None:
        beaches = df['Client.ID'].dropna().unique().tolist()

    if subplots:
        try:
            if len(subplots) != 2:
                raise ValueError('subplots must have exactly 2 elements')
        except TypeError:
            raise TypeError('subplots must be an iterable with 2 elements')

        if subplots[0] * subplots[1] < len(beaches):
            raise ValueError('not enough subplots for each beach')

        min_x = np.inf
        max_x = -np.inf
        for b in beaches:
            data = df[df['Client.ID'] == b][col].map(transform)
            if data.min() < min_x and not np.isinf(data.min()):
                min_x = data.min()
            if data.max() > max_x and not np.isinf(data.min()):
                max_x = data.max()

        fig, ax = plt.subplots(subplots[0], subplots[1],
                               sharex=True, sharey=True)
        ax = ax.flatten()

        for i, b in enumerate(beaches):
            df[df['Client.ID'] == b][col].map(transform).hist(
                normed=1, ax=ax[i], bins=np.linspace(min_x, max_x, 15)
            )
            ax[i].set_ylabel(b)
            ax[i].set_yticklabels([])

        for i in range(len(beaches) + 1, len(ax)):
            ax[i].set_yticklabels([])

    else:
        fig, ax = plt.subplots(1)
        for b in beaches:
            df[df['Client.ID'] == b][col].map(transform).hist(
                normed=True, alpha=.5, ax=ax
            )
        ax.legend(beaches)

    plt.show(block=TO_BLOCK)


def movie(compare_column=None, df=None):
    '''
    Creates an animation of the beaches E. coli levels represented as circles.
    The circle's radius is proportional to the log of the E. coli levels.
    Additionally, when the E. coli level is above the threshold of 235 PPM,
    the circle color changes from blue to purple. You can optionally choose
    to vary the background color of the animation with another column of data,
    however, this does not seem like a great way to visualize the relationship
    between E. coli levels and another data-stream.

    Inputs
    ------
    compare_column : The name or index of the column that will be used to vary
                     the background color. If compare_column is None, then the
                     background color will remain static.
    df             : The dataframe to use. If None, then the dataframe will be
                     read in using read_data.

    Returns
    -------
    anim : The animation object.

    Example
    -------
    >>> import read_data as rd
    >>> import visualizations as viz
    >>> df = rd.read_data()
    >>> viz.movie(df=df)
    '''

    if df is None:
        df = read_data.read_data()
    if compare_column is None:
        to_compare = False
    else:
        to_compare = True

    if to_compare:
        compare_min = df[compare_column].dropna().min()
        compare_max = df[compare_column].dropna().max()
        bg_min_color = np.array([.75, .5, .2])
        bg_max_color = np.array([.999, .999, 0.9])

    file_name = '../data/ExternalData/Beach_Locations.csv'
    beach_locs = read_data.read_locations(file_name)

    # compute Mercator projection of lat/longs
    phi = 0.730191653

    beach_locs['Latitude'] = beach_locs['Latitude'] * 110574.0
    beach_locs['Longitude'] = beach_locs['Longitude'] * 111320.0 * np.cos(phi)

    lat_min = beach_locs['Latitude'].min()
    lat_max = beach_locs['Latitude'].max()
    lat_rng = lat_max - lat_min
    lon_min = beach_locs['Longitude'].min()
    lon_max = beach_locs['Longitude'].max()
    lon_rng = lon_max - lon_min

    def generate_index():
        for timestamp in df.index.unique():
            readings = df.ix[timestamp, 'Escherichia.coli']
            if to_compare:
                compare = df.ix[timestamp, compare_column]
                if type(compare) is pd.Series:
                    compare = compare.dropna().mean()
                if np.isnan(compare):
                    continue
            if ((type(readings) is np.float64 and not np.isnan(readings)) or
                    (type(readings) is not np.float64 and readings.count())):
                if not to_compare:
                    compare = None
                yield timestamp, compare

    def animate(timestamp_and_compare):
        timestamp = timestamp_and_compare[0]
        compare = timestamp_and_compare[1]

        if to_compare:
            compare = (compare - compare_min) / compare_max
            bg_color = bg_min_color * compare + bg_max_color * (1. - compare)
            ax.set_axis_bgcolor(bg_color)

        for i, b in enumerate(beach_locs['Beach']):
            beach_filt = df.ix[timestamp, 'Client.ID'] == b
            beach_skipped = False
            try:
                if not beach_filt.sum() == 1:
                    beach_skipped = True
            except AttributeError:  # is a boolean
                if not beach_filt:
                    beach_skipped = True

            if beach_skipped:
                ecoli = 0
            else:
                ecoli = float(df.ix[timestamp, 'Escherichia.coli'][beach_filt])

            r = 200 * np.log(ecoli)

            if b in circle_indexes:
                ax.artists[circle_indexes[b]].set_radius(r)
                if ecoli >= 235:
                    ax.artists[circle_indexes[b]].set_facecolor(
                        (0.301, 0, 1, 0.75))
                else:
                    ax.artists[circle_indexes[b]].set_facecolor(
                        (0, 0.682, 1, 0.75))
            else:
                circ = plt.Circle((beach_locs.ix[i,'Longitude'],
                                   beach_locs.ix[i,'Latitude']),
                                  radius=r, edgecolor='none')
                ax.add_artist(circ)
                circle_indexes[b] = len(ax.artists) - 1
                if ecoli >= 235:
                    ax.artists[circle_indexes[b]].set_facecolor(
                        (0.301, 0, 1, 0.75))
                else:
                    ax.artists[circle_indexes[b]].set_facecolor(
                        (0, 0.682, 1, 0.75))
        ax.title.set_text(timestamp.strftime('%d %B %Y'))
        return ax

    fig = plt.figure(figsize=(18,10))
    ax = plt.gcf().gca()
    ax.set_xlim([lon_min - lon_rng * 0.4, lon_max + lon_rng * 0.15])
    ax.set_ylim([lat_min - lat_rng * 0.2, lat_max + lat_rng * 0.2])
    ax.set_aspect('equal')
    circle_indexes = {}

    anim = animation.FuncAnimation(fig, animate, generate_index, repeat=False)
    plt.show(block=TO_BLOCK)

    return anim


def plot_beach(columns, df=None, beaches=None, separate_beaches=False, **kwds):
    '''
    Plots the specified column of data for the specified beaches.

    Inputs
    ------
    columns          : One or more column names/indexes of data to plot.
    df               : The dataframe of data. If None, then the dataframe
                       will be read in using read_data.
    beaches          : Name or list of names of beaches to plot. If None, all
                       beaches will be used.
    separate_beaches : If False, each beach will be plotted on the same axis.
                       Otherwise, each beach will be plotted on its own axis.

    keyword arguments
    -----------------
    Other keyword arguments will be past to the plot routine.

    Returns
    fig : The figure object.
    ax  : If separate_beaches is false, then this is the axis object.
          Otherwise, it is the array of axis objects.

    Example
    -------
    >>> import read_data as rd
    >>> import visualizations as viz
    >>> df = rd.read_data()
    >>> beaches = ['Juneway', 'Rogers', 'Howard']
    >>> col = 'Escherichia.coli'
    >>> viz.plot_beach(col, df=df, beaches=beaches, separate_beaches=True)
    '''
    if df is None:
        df = read_data.read_data()
    if beaches is None:
        beaches = df['Client.ID'].dropna().unique().tolist()
    if type(beaches) is str:
        # be flexible with scalar vs. vector input
        beaches = [beaches]

    if separate_beaches:
        fig, ax = plt.subplots(len(beaches), 1, sharex=True, sharey=True)
        for i, beach in enumerate(beaches):
            filt = df['Client.ID'] == beach
            df[filt].plot(y=columns, ax=ax[i], **kwds)
            ax[i].set_title(beach)
    else:
        fig, ax = plt.subplots(1,1)
        for i, beach in enumerate(beaches):
            if type(columns) is str:
                l = i
            else:
                l = i * len(columns)
            filt = df['Client.ID'] == beach
            df[filt].plot(y=columns, ax=ax, **kwds)
            # TODO: cannot get this legend stuff to work...
            for txt in ax.legend().get_texts()[l:]:
                txt.set_text(beach + ': ' + txt.get_text())

    plt.show(block=TO_BLOCK)

    return fig, ax
