import read_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def movie(data_column, lat_longs, df=None):

    if df is None:
        df = read_data.read_data()

    file_name = '../data/ExternalData/Beach_Locations.csv'
    beach_locs = read_data.read_locations(file_name)

    phi = 0.730191653

    beach_locs['Latitude'] = beach_locs['Latitude'] * 110574.0
    beach_locs['Longitude'] = beach_locs['Longitude'] * 111320.0 * np.cos(phi)

    # lat_longs = np.array(lat_longs)
    # lat_longs[:,0] = lat_longs[:,0] * 110574.0
    # lat_longs[:,1] = lat_longs[:,1] * 111320.0 * np.cos(phi)

    # fig = plt.gcf()
    # ax = fig.gca()
    lat_min = beach_locs['Latitude'].min()
    lat_max = beach_locs['Latitude'].max()
    lat_rng = lat_max - lat_min
    lon_min = beach_locs['Longitude'].min()
    lon_max = beach_locs['Longitude'].max()
    lon_rng = lon_max - lon_min
    # ax.set_xlim([lon_min - lon_rng * 0.1, lon_max + lon_rng * 0.1])
    # ax.set_ylim([lat_min - lat_rng * 0.1, lat_max + lat_rng * 0.1])
    # ax.set_aspect('equal')

    def generate_index():
        for timestamp in df.index.unique():
            readings = df.ix[timestamp, 'Escherichia.coli']
            if ((type(readings) is np.float64 and not np.isnan(readings)) or
                    (type(readings) is not np.float64 and readings.count())):
                yield timestamp

    def animate(timestamp):
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
                    ax.artists[circle_indexes[b]].set_facecolor((0.862, 0.357, 0.276, 0.8))
                else:
                    ax.artists[circle_indexes[b]].set_facecolor((0.262, 0.357, 0.576, 0.8))
            else:
                circ = plt.Circle((beach_locs.ix[i,'Longitude'],
                                   beach_locs.ix[i,'Latitude']),
                                  radius=r)
                ax.add_artist(circ)
                circle_indexes[b] = len(ax.artists) - 1
                if ecoli >= 235:
                    ax.artists[circle_indexes[b]].set_facecolor((0.862, 0.357, 0.276, 0.8))
                else:
                    ax.artists[circle_indexes[b]].set_facecolor((0.262, 0.357, 0.576, 0.8))
        ax.title.set_text(timestamp.strftime('%d %B %Y'))
        return ax

    fig = plt.figure(figsize=(18,10))
    ax = plt.gcf().gca()
    ax.set_xlim([lon_min - lon_rng * 0.4, lon_max + lon_rng * 0.15])
    ax.set_ylim([lat_min - lat_rng * 0.2, lat_max + lat_rng * 0.2])
    ax.set_aspect('equal')
    circle_indexes = {}

    anim = animation.FuncAnimation(fig, animate, generate_index, repeat=False)

    mywriter = animation.FFMpegWriter(fps=30)
    anim.save('test.mp4', writer=mywriter)

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
    return fig, ax


if __name__ == '__main__':
    df = read_data.read_data()

    plt.show()
