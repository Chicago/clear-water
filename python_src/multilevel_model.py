import numpy as np
import read_data as rd
import matplotlib.pyplot as plt
import pymc as pm
import re
import visualizations as viz


class multilevel_model:
    '''
    Creates the following model.

    Let Y be the binary decision whether or not E. coli levels are elevated.
    Xj be the j'th predictor in the model. Then we assume that if Y_ib is
    the i'th observation at the b'th beach, then

    Y_ib ~ Bernoulli(p_hat_ib)

    p_hat_ib = INVERSE LOGIT(SUM OVER j(Xj_ib * beta_jb))

    beta_jb ~ N(mu_beta, sigma_beta)

    mu_b ~ N(0,100^2)

    sigma_beta ~ N(0,100^2)
    '''
    def __init__(self, columns=None, df=None, alpha=None, beta=None):
        if df is None:
            df = rd.read_data()
        if columns is None:
            columns = [
                'Reading.1'
            ]

        df = df[['Client.ID','Full_date','Escherichia.coli'] + columns]
        df = df.dropna()
        self.unique_beaches = df['Client.ID'].unique()
        self.beach_indexes = np.array(
            df['Client.ID'].map(lambda x: self.unique_beaches.tolist().index(x))
        )
        self.X = np.array(df[columns],)
        # TODO: handle one dim data
        self.X = np.concatenate((self.X, np.ones([self.X.shape[0], 1])), 1)
        self.columns = columns + ['Constant']
        self.Y = np.array(df['Escherichia.coli'] > 235, dtype=np.float)
        self.train_indexes = (df['Full_date'] < '1-1-2015').values
        self.run_model()
        self.get_performance()

    def run_model(self):
        X = self.X[self.train_indexes,:]
        Y_obs = self.Y[self.train_indexes]
        beach_indexes = self.beach_indexes[self.train_indexes]

        M = X.shape[1]
        num_beaches = len(self.unique_beaches)

        # define priors
        mu_betas = pm.Normal('mu_beta', mu=0, tau=1.0 / 100**2, size=M, value=np.zeros(M))
        sigma_betas = pm.HalfNormal('sigma_beta', tau=1. / 100.0**2, size=M)
        betas = pm.Container([pm.Normal(self.columns[i] + '_coef',
                                        mu=mu_betas[i],
                                        tau=1. / sigma_betas[i] ** 2.0,
                                        size=num_beaches,
                                        value=np.zeros(num_beaches))
                             for i in range(M)])

        # define estimate
        @pm.deterministic
        def p_est(betas=betas, x=X):
            est = np.zeros(len(beach_indexes))
            for i in range(M):
                est = est + betas[i][beach_indexes] * x[:,i]
            return pm.invlogit(est)

        @pm.observed
        def p_hat(p=p_est, value=Y_obs):
            return pm.bernoulli_like(value, p)

        # inference
        self.model = pm.Model([mu_betas, sigma_betas, betas, p_est, p_hat])
        self.mc = pm.MCMC(self.model)
        self.mc.sample(iter=500000, burn=250000, thin=200)

        # # display info
        # bbar = map(lambda x: x.stats()['mean'], betas)
        # for mean_values in bbar:
        #     print(mean_values)
        # pm.Matplot.plot(self.mc, common_scale=False)
        # for i in range(M):
        #     pm.Matplot.plot(betas[i])
        # plt.show(block=False)

        self.betas = betas

    def get_performance(self):
        X = self.X[~self.train_indexes,:]
        Y = self.Y[~self.train_indexes]
        beach_indexes = self.beach_indexes[~self.train_indexes]

        M = X.shape[1]

        bbar = map(lambda x: x.stats()['mean'], self.betas)

        est = np.zeros(len(beach_indexes))
        for i in range(M):
            est = est + bbar[i][beach_indexes] * X[:,i]
        est = pm.invlogit(est)

        viz.roc(est, Y)
        viz.precision_recall(est, Y)


def main():
    # Load data, don't load water/weather station data
    df = rd.read_data(read_water_sensor=False, read_weather_station=False)

    # Specify the variables to use
    numeric_model_variables = [
        # 'Escherichia.coli',
        # 'precipIntensity',
        'precipIntensityMax',
        'precipProbability',
        'temperatureMin',
        'temperatureMax',
        # 'humidity',
        'windSpeed',
        # 'windBearing',
    ]
    categorical_model_variables = [
        # 'icon',
    ]

    # get historical data
    model_variables = numeric_model_variables + categorical_model_variables
    lookback_smoothing_days = 7
    df = rd.add_column_prior_data(df, model_variables, range(1,lookback_smoothing_days+1),
                                  beach_col_name='Client.ID', timestamp_col_name='Full_date')

    columns_to_use = []
    lookback_model_days = 2
    r = re.compile('(^[0-9]+)_day_prior_(.*)')
    df_cols = df.columns
    for c in df_cols:
        match = r.match(c)
        if match:
            if int(match.group(1)) <= lookback_model_days:
                columns_to_use.append(c)
        else:
            priors = df_cols.map(lambda x: (r.match(x) is not None) and r.match(x).group(2) == c)
            prior_cols = df_cols[priors].tolist()
            if not len(prior_cols):
                continue
            df[prior_cols] = df[['Client.ID'] + prior_cols].groupby('Client.ID').transform(
                lambda x: x.fillna(x.mean())
            )

    m = multilevel_model(df=df, columns=columns_to_use)

    return m, df


if __name__ == '__main__':
    main()
