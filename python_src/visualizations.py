import read_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def roc(scores, labels):
    '''
    Plots the Receiver Operator Characteristic (ROC) curve of the results
    from a binary classification system.

    Inputs
    ------
    scores : Nx1 array of (usually continuous) outputs of a classification
             system, for example, predicted E. coli levels
    labels : Nx1 boolean array of true classes

    Returns
    -------
    fpr      : Mx1 array of false positive rates
    tpr      : Mx1 array of true positive rates
    threshes : Mx1 array of the thresholds on the scores variable
               used to create the corresponding fpr and tpr values.

    Example
    -------
    import read_data as rd
    df = rd.read_data()
    scores = df[['Reading.1', 'Escherichia.coli']].dropna()['Reading.1']
    labels = df[['Reading.1', 'Escherichia.coli']].dropna()['Escherichia.coli']
    labels = labels >= 235.0
    roc(scores, labels)
    # Warning! This will perform very well b/c it's predcting today
    # using today's data, this is not a viable model!
    '''
    scores = np.array(scores)
    labels = np.array(labels)

    sort_inds = np.argsort(scores)[::-1]
    scores = scores[sort_inds]
    labels = labels[sort_inds]

    labels = labels.astype('bool')

    # Adapted from sklearn.metrics._binary_clf_curve:
    # scores typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # We need to use isclose to avoid spurious repeated thresholds
    # stemming from floating point roundoff errors.
    distinct_value_indices = np.where(np.logical_not(np.abs(
        np.diff(scores)) < 0.00001))[0]
    threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = labels.cumsum()[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tpr = tps / float(tps[-1])
    fpr = fps / float(fps[-1])

    fig, ax = plt.subplots(1)
    ax.plot(fpr, tpr)
    ax.hold(True)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')

    plt.show()

    return fpr, tpr, scores[threshold_idxs]


def precision_recall(scores, labels):
    '''
    Plots the Precision Recall (PR) curve of the results
    from a binary classification system.

    Inputs
    ------
    scores : Nx1 array of (usually continuous) outputs of a classification
             system, for example, predicted E. coli levels
    labels : Nx1 boolean array of true classes

    Returns
    -------
    tpr      : Mx1 array of true positive rates
    ppv      : Mx1 array of positive predictive values
    threshes : Mx1 array of the thresholds on the scores variable
               used to create the corresponding ppv and tpr values.

    Example
    -------
    import read_data as rd
    df = rd.read_data()
    scores = df[['Reading.1', 'Escherichia.coli']].dropna()['Reading.1']
    labels = df[['Reading.1', 'Escherichia.coli']].dropna()['Escherichia.coli']
    labels = labels >= 235.0
    precision_recall(scores, labels)
    # Warning! This will perform very well b/c it's predcting today
    # using today's data, this is not a viable model!
    '''
    scores = np.array(scores)
    labels = np.array(labels)

    sort_inds = np.argsort(scores)[::-1]
    scores = scores[sort_inds]
    labels = labels[sort_inds]

    labels = labels.astype('bool')

    # Adapted from sklearn.metrics._binary_clf_curve:
    # scores typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # We need to use isclose to avoid spurious repeated thresholds
    # stemming from floating point roundoff errors.
    distinct_value_indices = np.where(np.logical_not(
        np.abs(np.diff(scores)) < 0.00001))[0]
    threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

    ppv = np.zeros(threshold_idxs.size)
    tpr = np.zeros(threshold_idxs.size)
    for i, thresh in enumerate(scores[threshold_idxs]):
        predict_pos = (scores >= thresh)
        ppv[i] = (predict_pos & labels).sum() / float(predict_pos.sum())
        tpr[i] = (predict_pos & labels).sum() / float(labels.sum())

    fig, ax = plt.subplots(1)
    ax.plot(tpr, ppv)
    ax.hold(True)
    ax.plot([0, 1], [float(labels.sum()) / labels.size,
                     float(labels.sum()) / labels.size], 'r--')
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('Positive Predictive Value')
    ax.set_title('PR curve')

    plt.show()

    return tpr, ppv, scores[threshold_idxs]


def beach_hist(col='Escherichia.coli', beaches=None,
               subplots=False, transform=lambda x: x, df=None):
    '''
    TODO: docstring
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
                normed=1, ax=ax[i], bins=np.linspace(min_x, max_x, 11)
            )
            ax[i].set_ylabel(b)
            ax[i].set_yticklabels([])

        for i in range(len(beaches) + 1, len(ax)):
            ax[i].set_yticklabels([])

    else:
        fig, ax = plt.subplots(1)
        for b in beaches:
            df[df['Client.ID'] == b][col].map(transform).hist(
                normed=1, alpha=.5, ax=ax
            )
        ax.legend(beaches)

    plt.show()


def movie(compare_column, df=None):
    '''
    TODO: docstring
    '''

    if df is None:
        df = read_data.read_data()

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
            compare = df.ix[timestamp, compare_column]
            if type(compare) is pd.Series:
                compare = compare.dropna().mean()
            if np.isnan(compare):
                continue
            if ((type(readings) is np.float64 and not np.isnan(readings)) or
                    (type(readings) is not np.float64 and readings.count())):
                yield timestamp, compare

    def animate(timestamp_and_compare):
        timestamp = timestamp_and_compare[0]
        compare = timestamp_and_compare[1]

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
    plt.show()

    return anim


def plot_beach(columns, df=None, beaches=None, separate_beaches=False, **kwds):
    '''
    TODO: docstring
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
        l = len(ax.legend().get_texts())
        for beach in beaches:
            filt = df['Client.ID'] == beach
            df[filt].plot(y=columns, ax=ax, **kwds)
            for txt in ax.legend().get_texts()[l:]:
                txt.set_text(beach + ': ' + txt.get_text())

    plt.show()

    return fig, ax


if __name__ == '__main__':
    df = read_data.read_data()

    scores = df[['Reading.1', 'Escherichia.coli']].dropna()['Reading.1']
    labels = df[['Reading.1', 'Escherichia.coli']].dropna()['Escherichia.coli']
    labels = labels >= 235.0
    roc(scores, labels)
    precision_recall(scores, labels)
