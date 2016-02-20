import numpy as np
import read_data as rd
import scipy.stats as sst
import matplotlib.pyplot as plt
import pymc as pm


class multilevel_model:
    '''
    Creates the following model.

    TODO
    '''
    def __init__(self, columns=None, df=None, alpha=None, beta=None):
        if df is None:
            df = rd.read_data()
        if columns is None:
            columns = [
                'Reading.1'
            ]

        df = df[['Client.ID','Escherichia.coli'] + columns]
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
        self.run_model()

    def run_model(self):
        X = self.X
        Y_obs = self.Y

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
        def p_est(betas=betas, x=self.X):
            est = np.zeros(len(self.beach_indexes))
            for i in range(M):
                est = est + betas[i][self.beach_indexes] * x[:,i]
            return pm.invlogit(est)

        @pm.observed
        def p_hat(p=p_est, value=Y_obs):
            return pm.bernoulli_like(value, p)

        # inference
        self.model = pm.Model([mu_betas, sigma_betas, betas, p_est, p_hat])
        self.mc = pm.MCMC(self.model)
        self.mc.sample(iter=500000, burn=200000, thin=100)

        bbar = map(lambda x: x.stats()['mean'], betas)
        for mean_values in bbar:
            print(mean_values)
        pm.Matplot.plot(self.mc, common_scale=False)
        # for i in range(M):
        #     pm.Matplot.plot(betas[i])
        plt.show(block=False)

if __name__ == '__main__':
    multilevel_model()
