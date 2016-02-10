import numpy as np
import pandas as pd
import read_data as rd
import random


def multilevel_linear_model(columns, df=None):
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
    '''
    if df is None:
        df = rd.read_data()

    def norm(mu, sigma, N):
        return np.random.normal(mu, sigma, N)

    def gibbs(niter, burnin, y, t, alpha, gamma, delta, N):
        # hyper-parameters
        mu_alphas = np.zeros(niter - burnin)
        sigma_alphas = np.zeros(niter - burnin)
        mu_betas = np.zeros(niter - burnin)
        sigma_betas = np.zeros(niter - burnin)
        # parameters
        alphas = np.zeros([niter - burnin, N])
        betas = np.zeros([niter - burnin, N])
        epsilon = np.zeros(niter - burnin)

        for i in range(niter):
            1

    return 1
