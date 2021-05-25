import numpy as np 
from scipy.stats import multivariate_normal


def response(x, pi, mu, sigma):
    ''' Calculate responsibility values

    Args:
        x: <np.array> [sample_number * dimension] samples
        pi: <np.array> [cluster_number] weight for gaussians
        mu: <np.array> [cluster_number * dimension] mean for gaussians
        sigma: <np.array> [cluster_number * dimension] variance for gaussians
    
    Returns:
        r: <np.array> [sample_number * cluster_number] responsibility values 
    '''
    pdfs = np.empty((len(x), len(pi)))
    for i in range(len(pi)):
        for j in range(x.shape[0]):
            pdfs[j, i] = pi[i] * multivariate_normal.pdf(x[j], mu[i], sigma[i])
    r = pdfs / pdfs.sum(axis=1).reshape(-1, 1)   
    return r 


def update_pi(r):
    ''' Update weight for gaussians
    
    Args: 
        r: <np.array> [sample_number * cluster_number] responsibility values 

    Returns:
        pi: <np.array> [cluster_number] weight for gaussians
    '''
    pi = r.sum(axis=0) / r.shape[0]
    return pi


def update_mu(x, r):
    ''' Update the mean for gaussians
    
    Args:
        x: <np.array> [sample_number * dimension] samples
        r: <np.array> [sample_number * cluster_number] responsibility values 
    
    Returns:
        mu: <np.array> [cluster_number * dimension] mean for gaussians
    '''
    k = r.shape[1]
    dim = x.shape[1]
    mu = np.empty((k, dim))
    r_sum = r.sum(axis=0)
    for i in range(k):
        mu[i] = np.multiply(r[:, i].reshape(-1, 1), x).sum(axis=0) / r_sum[i]
    return mu


def update_sigma(x, r, mu):
    ''' Update the variance for gaussians

    Args:
        x: <np.array> [sample_number * dimension] samples
        r: <np.array> [sample_number * cluster_number] responsibility values 
        mu: <np.array> [cluster_number * dimension] mean for gaussians
    
    Returns:
        sigma: <np.array> [cluster_number * dimension] variance for gaussians
    '''
    n = x.shape[0]
    k = r.shape[1]
    dim = x.shape[1]
    sigma = np.empty((k, dim, dim))
    r_sum = r.sum(axis=0)
    for i in range(k):
        temp = np.zeros((dim, dim))
        for j in range(n):
            temp += r[j, i] * np.outer((x[j] - mu[i]).T, x[j] - mu[i]) 
        sigma[i] = temp / r_sum[i]
    return sigma


def log_likelihood(x, pi, mu, sigma):
    ''' Calculate the log likelihood for samples

    Args:
        x: <np.array> [sample_number * dimension] samples
        pi: <np.array> [cluster_number] weight for gaussians
        mu: <np.array> [cluster_number * dimension] mean for gaussians
        sigma: <np.array> [cluster_number * dimension] variance for gaussians

    Returns:
        likelihood: <float> calculated likelihood
    '''
    n = x.shape[0]
    k = pi.shape[0]
    pdfs = np.zeros((n, k))
    for i in range(k):
        for j in range(x.shape[0]):
            pdfs[j, i] = pi[i] * multivariate_normal.pdf(x[j], mu[i], sigma[i])
    likelihood = np.mean(np.log(pdfs.sum(axis=1)))
    return likelihood


def mean(x, k):
    '''Calculate the midpoint of datas

    Args:
        x: <np.array> [sample_number * dimension] samples
        k: <float> [cluster_number] cluster_number

    Returns:
        mu: <np.array> [cluster_number * dimension] mean for gaussians
    '''
    min_x = np.min(x[:, 0])
    max_x = np.max(x[:, 0])
    list_x = np.linspace(min_x+ 0.2 * (max_x-min_x), max_x-0.2 * (max_x-min_x), 5)

    min_y = np.min(x[:, 1])
    max_y = np.max(x[:, 1])
    list_y = np.linspace(min_y+ 0.2 * (max_y-min_y), max_y-0.2 * (max_y-min_y), 5)

    mu = []
    for i in range(k):
        mu.append([list_x[i], list_y[i]])
    mu = np.array(mu)
    return mu