import numpy as np
import read_data as rd
import scipy.stats as sst
import matplotlib.pyplot as plt


class multilevel_model:
    '''
    Creates the following model.

    Variables:
        Y_ib is the true status of E. coli (binary), and is the
        i'th reading at beach b.

        alpha_b is the intercept for beach b

        beta_b is the coefficient for beach b

        epsilon is the error term of our prediction

        mu_alpha, sigma_alpha are hyper-parameters for the alphas

        mu_beta, sigma_beta are hyper-parameters for the betas

    Model:
        Y_ib ~ N(alpha_b + beta_b * X_ib, epsilon ** 2)

        alpha_b ~ N(mu_alpha, sigma_alpha ** 2)

        beta_b ~ N(mu_beta, sigma_beta ** 2)

        mu_alpha, sigma_alpha, mu_beta, sigma_beta ~ N(0, 100)

        epsilon ~ U(0,100)
    '''
    def __init__(self, columns=None, df=None, alpha=None, beta=None):
        if df is None:
            df = rd.read_data()
        if columns is None:
            columns = [
                'temperatureMax'
            ]

        df = df[['Client.ID','Escherichia.coli'] + columns]
        df = df.dropna()
        self.unique_beaches = df['Client.ID'].unique()
        self.beach_indexes = np.array(
            df['Client.ID'].map(lambda x: self.unique_beaches.tolist().index(x))
        )
        self.X = np.array(df[columns])
        self.X = np.append(self.X, np.ones([self.X.shape[0], 1]))
        self.Y = np.array(df['Escherichia.coli'])
        self.model()

    def model(self):
        X = self.X
        Y = self.Y

        M = X.shape[1]

        beta = np.ones(self.X.shape[0])

        # define priors
        mu_beta = pymc.Normal('mu_beta', mu=0, tau=1.0 / 100**2)
        sigma_beta = pymc.Uniform('tau_beta', low=1.0, hi=1.0)
        beta = pymc.Normal('coefficients', mu=mu_beta, tau=sigma_beta, size=M)
        epsilon = pymc.Gamma('epsilon', alpha=1.0, beta=1.0)

        # define estimate
        @pymc.deterministic
        def mu(beta=beta, x=x):
            est = alpha[self.beach_indexes] + np.sum(beta[self.beach_indexes,:] * self.X, 1)
            return a * x + b

        y = pymc.Normal('y', mu=mu, tau=tau, value=y_obs, observed=True)

        # inference
        m = pymc.Model([a, b, tau, x, y])
        mc = pymc.MCMC(m)
        mc.sample(iter=15000, burn=10000, thin=5)

        abar = a.stats()['mean']
        bbar = b.stats()['mean']
        data.plot(x='x', y='y', kind='scatter', s=50)
        xp = np.array([x.min(), x.max()])

        lines = a.trace() * xp[:, None] + b.trace()
        plt.plot(xp, lines, c='red', alpha=0.01)
        plt.plot(xp, abar * xp + bbar, linewidth=2, c='red')

        plt.figure()
        plt.plot(a.trace())
        plt.figure()
        plt.plot(b.trace())

        plt.show(block=False)
        print('')
        print(abar)
        print(bbar)


    def calc_prob(self, alpha, beta, eps, mu_alpha,
                  sigma_alpha, mu_beta, sigma_beta):
        est = alpha[self.beach_indexes] + np.sum(beta[self.beach_indexes,:] * self.X, 1)
        prob_X_given_model = np.sum(
            sst.norm.logpdf(self.Y, est, eps)
        )
        prob_model = np.sum(
            sst.norm.logpdf(alpha, mu_alpha, sigma_alpha)
        ) + np.sum(
            sst.norm.logpdf(beta, mu_beta, sigma_beta)
        ) + sst.norm.logpdf(mu_alpha, 0, 1000) + sst.norm.logpdf(mu_beta, 0, 1000)
        return prob_X_given_model + prob_model

    def metropolis_hastings(self, niter, burnin, step_size, step_width=2,
                            alpha=None, beta=None, eps=None):
        X = self.X
        N = X.shape[1]
        M = len(self.unique_beaches)

        if alpha is None:
            curr_alpha = np.zeros(M)
        else:
            curr_alpha = alpha
            # TODO: check dims

        if beta is None:
            curr_beta = np.ones([M, N])
        else:
            curr_beta = beta
            # TODO: check dims

        if eps is None:
            curr_eps = 50.
        else:
            curr_eps = eps
            # TODO: check dims

        curr_mu_alpha = np.random.normal(0, 100)
        curr_sigma_alpha = np.random.uniform(0, 100)
        curr_mu_beta = np.random.normal(0, 100)
        curr_sigma_beta = np.random.uniform(0, 100)

        curr_prob = self.calc_prob(curr_alpha, curr_beta, curr_eps, curr_mu_alpha,
                                   curr_sigma_alpha, curr_mu_beta, curr_sigma_beta)

        # hyper-parameters
        mu_alphas = np.zeros(niter)
        sigma_alphas = np.zeros(niter)
        mu_betas = np.zeros(niter)
        sigma_betas = np.zeros(niter)
        # parameters
        alphas = np.zeros([niter, M])
        betas = np.zeros([niter, M, N])
        epsilons = np.zeros(niter)

        idx = 0
        self.rhos = []
        self.curr_probs = []
        self.next_probs = []
        for i in range(niter * step_size + burnin):
            next_mu_alpha = np.random.normal(0, 100)
            next_sigma_alpha = np.random.uniform(0, 100)
            next_mu_beta = np.random.normal(0, 100)
            next_sigma_beta = np.random.uniform(0, 100)

            next_alpha = np.random.normal(next_mu_alpha, next_sigma_alpha, M)
            next_beta = np.random.normal(next_mu_beta, next_sigma_beta, [M, N])
            next_eps = np.random.uniform(0, 100)

            next_prob = self.calc_prob(next_alpha, next_beta, next_eps, next_mu_alpha,
                                       next_sigma_alpha, next_mu_beta, next_sigma_beta)

            rho = np.exp(next_prob) / np.exp(curr_prob)
            self.rhos.append(rho)
            self.curr_probs.append(curr_prob)
            self.next_probs.append(next_prob)

            if rho > 1 or rho > np.random.uniform():
                curr_prob = next_prob

                curr_mu_alpha = next_mu_alpha
                curr_sigma_alpha = next_sigma_alpha
                curr_mu_beta = next_mu_beta
                curr_sigma_beta = next_sigma_beta

                curr_alpha = next_alpha
                curr_beta = next_beta
                curr_eps = next_eps

            if i >= burnin and not i - burnin % step_size:
                mu_alphas[idx] = curr_mu_alpha
                sigma_alphas[idx] = curr_sigma_alpha
                mu_betas[idx] = curr_mu_beta
                sigma_betas[idx] = curr_sigma_beta

                alphas[idx,:] = curr_alpha
                betas[idx,:,:] = curr_beta
                epsilons[idx] = curr_eps

        self.mu_alphas = mu_alphas
        self.sigma_alphas = sigma_alphas
        self.mu_betas = mu_betas
        self.sigma_betas = sigma_betas

        self.alphas = alphas
        self.betas = betas
        self.epsilons = epsilons

        plt.plot(self.rhos)
        plt.show()

if __name__ == '__main__':
    m = multilevel_model()
