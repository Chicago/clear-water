import read_data as rd
import datetime as dt
import visualizations as viz
import matplotlib.pyplot as plt
import numpy as np


def check_sample_times(df=None, to_plot=False):
    '''
    Investigates whether there is a relationship between the time a
    sample was taken and the E. coli reading. A possible hypothesis
    being that samples taken later in the day might tend to read be
    higher.

    The conclusions from this function seem to indicate that there
    is not a subtantial relationship between sample time and E.
    coli reading.

    Inputs
    ------
    df      : Dataframe object, should contain at least the columns
              'Client.ID', 'Escherichia.coli', 'Sample.Collection.Time',
              if df is None, then it will be read in from read_data.
    to_plot : Boolean, if true, the results will be printed and
              plotted. Otherwise, just the cleansed dataframe will
              be returned.

    Returns
    -------
    ct : Dataframe of collection times and E. coli readings.
         The column 'Sample.Collection.Time' is the fraction of the day,
         for example, a value of 0.50 indicates the collection happened
         at noon, a value of 0.25 would indicate 6:00 AM, etc.
    '''
    if df is None:
        df = rd.read_data()

    ct = df[['Client.ID', 'Escherichia.coli', 'Sample.Collection.Time']].dropna()

    def clean_times(s):
        '''
        Takes in a string from the sample collection column and
        makes it machine readable if possible, and a NaN otherwise
        '''
        if type(s) is not str:
            if type(s) is dt.datetime or type(s) is dt.time:
                return dt.datetime(2016, 1, 1, hour=s.hour, minute=s.minute)

        try:
            if ':' not in s:
                return float('nan')
            i = s.index(':')
            hr = int(s[max(i - 2, 0):i])
            mn = int(s[i+1:i+3])

            return dt.datetime(2016, 1, 1, hour=hr, minute=mn)
        except:
            return float('nan')

    ct['Sample.Collection.Time'] = ct['Sample.Collection.Time'].map(clean_times)
    ct = ct.dropna()
    ct['Sample.Collection.Time'] = ct['Sample.Collection.Time'].map(
        lambda x: x.hour / 24. + x.minute / (24. * 60.)
    )
    # Filter out those samples which came before 4:00 AM or after 8:00 PM
    ct = ct[(ct['Sample.Collection.Time'] > .125) & (ct['Sample.Collection.Time'] < .83)]

    if to_plot:
        ct['Escherichia.coli'] = ct['Escherichia.coli'].map(lambda x: np.log(x + 1.))
        print('Pearson correlation : ' + str(ct.corr(method='pearson').ix[0,1]))
        print('Spearman correlation: ' + str(ct.corr(method='spearman').ix[0,1]))
        ct.plot(x='Escherichia.coli', y='Sample.Collection.Time', style='.')
        ax = plt.gca()
        ax.set_xlim([ct['Escherichia.coli'].min(), ct['Escherichia.coli'].max()])
        ct.plot(y='Escherichia.coli', x='Sample.Collection.Time', style='.')
        ax = plt.gca()
        ax.set_xlim([ct['Sample.Collection.Time'].min(), ct['Sample.Collection.Time'].max()])
        tb = viz.TO_BLOCK
        viz.TO_BLOCK = False
        fig, ax = viz.plot_beach(columns='Sample.Collection.Time', df=ct)
        viz.TO_BLOCK = tb
        ax.legend_.remove()
        plt.show(tb)
        ct['Escherichia.coli'] = ct['Escherichia.coli'].map(lambda x: np.exp(x) - 1.)

    return ct


if __name__ == '__main__':
    check_sample_times(to_plot=True)
