
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: solution.ipynb

import numpy as np
from helper_functions import *

def get_initial_means(array, k):
    size = array.size
    y, x = array.shape[0], array.shape[1]
    initial_means = np.ndarray([k, x])
    r = np.random.choice(y, k, replace=False)
    for i in range(k):
        ppt = array[r[i]]
        initial_means[i] = ppt
    return initial_means

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_step(X, k, means=None):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and
    calculate new means.
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """

    def calc_dist(arr, one_pt):
        k_dists = np.power(arr - one_pt, 2)
        k_dists = np.sqrt(np.sum(k_dists, axis=1))
        return k_dists

    mean_dists = np.ndarray([X.shape[0], 0])
    for pt in means:
        # k_dists = np.linalg.norm(X - pt, axis=1)
        k_dists = calc_dist(X, pt)
        mean_dists = np.append(mean_dists, k_dists[:, np.newaxis], axis=1)

    # find the nearest mean for each feature
    mins = np.argmin(mean_dists, axis=1)
    mins = mins[:, np.newaxis]
    labeled_features = np.append(X, mins, axis=1)  # append nearest mean to list of features

    # place each feature into the correct cluster array
    # and calculate new means
    cluster_means = np.zeros([k, means.shape[1]])
    for i in range(k):
        labeled_cluster = labeled_features[np.where(labeled_features[:, -1] == i)]
        cluster_means[i, :] = labeled_cluster[:, 0:-1].mean(axis=0, dtype=np.float64)

    return cluster_means, labeled_features[:, -1]

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """
    X = flatten_image_matrix(image_values)
    if initial_means is None:
        rng = np.random.default_rng()
        numbers = rng.choice(X.shape[0], size=k, replace=False)
        initial_means = X[numbers]
    new_means, labeled_clusters = k_means_step(X, k, initial_means)

    count = 0
    while True:
        if count > 100:
            break
        count += 1
        last_means = new_means
        new_means, labeled_clusters = k_means_step(X, k, new_means)
        if abs(np.sum(last_means) - np.sum(new_means)) == 0:
            for i in range(k):
                # put ith mean into the labeled_clusters
                ind = np.argwhere(labeled_clusters == i)
                X[ind, :] = new_means[i]
            new_img = unflatten_image_matrix(X, image_values.shape[0])
            return new_img

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    m = X.shape[0]
    n = X.shape[1]
    k = MU.shape[0]
    SIGMA = np.zeros([k, n, n])
    for i in range(k):
        d = np.subtract(X.copy(), MU[i])
        cov_i = np.dot(d.T, d)
        # cov_i = d.T @ d
        cov_i = np.divide(cov_i, m)
        SIGMA[i, :, :] = cov_i

    return SIGMA

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    """
    m = X.shape[0]
    n = X.shape[1]
    MU = np.zeros([k, n])
    PI = np.zeros([k, 1])

    # initialize mu
    MU[:, :] = get_initial_means(X, k)

    # init sigma
    # SIGMA = np.zeros([k, n, n])
    SIGMA = compute_sigma(X, MU)

    # initialize pi
    PI[:, :] = 1 / k

    return MU, SIGMA, PI

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def prob(x, mu, sigma):
    """Calculate the probability of a single
    data point x under component with
    the given mean and covariance.
    # NOTE: there is nothing to vectorize here yet,
    # it's a simple check to make sure you got the
    # multivariate normal distribution formula right
    # which is given by N(x;MU,SIGMA) above

    params:
    x = numpy.ndarray[float]
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float
    """
    x_min_mu = np.subtract(x, mu)
    d = sigma.shape[0]
    x1 = np.divide(1, np.power(2 * np.pi, (np.divide(d, 2))))
    x2 = np.power(np.linalg.det(sigma), (-0.5))
    x3 = np.exp(np.multiply((-0.5), np.linalg.multi_dot([x_min_mu.T, np.linalg.inv(sigma), x_min_mu])))
    p = x1 * x2 * x3
    return p

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    sigma = SIGMA

    responsibility = np.zeros([k, X.shape[0]])

    for k in range(k):
        r = np.copy(X)
        mu_k = MU[k]
        sigma_k = sigma[k]
        try:
            invsig_k = np.linalg.inv(sigma_k)
        except np.linalg.LinAlgError:
            print("Inverse error for matrix: ", sigma_k)
            invsig_k = sigma_k
        x_sub_mus = np.subtract(r, mu_k)

        d = 3

        # X1: 1 / 2pi^d/2
        x1 = np.divide(1, np.power(2 * np.pi, (np.divide(d, 2))))

        # X2: sig^-1/2
        x2 = np.power(np.linalg.det(sigma_k), (-0.5))

        # X3: Tricky vectorization part, exp(-1/2*invsig*x-mu*x-mu.T)
        x3 = np.dot(invsig_k, x_sub_mus.T).T
        x3 = np.sum(x_sub_mus * x3, axis=1)
        x3 = np.exp(np.multiply((-0.5), x3))

        # Combine formula
        r = x1 * x2 * x3
        r = PI[k] * r

        responsibility[k, :] = r

    rows_sums = np.sum(responsibility, axis=0)
    responsibility = responsibility / rows_sums

    return responsibility


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    """
    m = X.shape[0]
    n = X.shape[1]
    new_MU = np.zeros([k, n])
    new_SIGMA = np.zeros([k, n, n])
    new_PI = np.zeros([k])

    r_row_sums = np.sum(r, axis=1)
    r_total = np.sum(r_row_sums)

    for i in range(k):
        N_c = r_row_sums[i]

        # PHI (verified)
        tmp_pi = N_c / r_total
        new_PI[i] = tmp_pi

        # MU
        r_ic = r[i]
        r_ic = np.expand_dims(r_ic, axis=1)
        mu_tmp = np.sum(X * r_ic, axis=0) / N_c
        new_MU[i, :] = mu_tmp

        # SIGMA
        r_ic = r[i]
        d = np.subtract(X.copy(), mu_tmp)
        cov_i = np.dot(r_ic * d.T, d)
        cov_i = np.divide(cov_i, N_c)
        new_SIGMA[i, :, :] = cov_i

    return new_MU, new_SIGMA, new_PI  # new_PI.T

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log10(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), log10(sum((k=1 to K),
                                      mixing_k * N(x_n | mean_k,stdev_k))))

    Make sure you are using log base 10, instead of log base 2.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    log_likelihood = float
    """
    responsibility = np.zeros([k, X.shape[0]])

    # print(SIGMA)
    # print(compute_sigma(X, MU))
    # SIGMA = compute_sigma(X, MU)

    for kk in range(k):
        r = np.copy(X)
        mu_k = MU[kk]

        sigma_k = SIGMA[kk]
        try:
            invsig_k = np.linalg.inv(sigma_k)
        except np.linalg.LinAlgError:
            print("Inverse error for matrix: ", sigma_k)
            invsig_k = sigma_k
        x_sub_mus = np.subtract(r, mu_k)

        d = sigma_k.shape[0]

        # X1: 1 / 2pi^d/2
        x1 = np.divide(1, np.power(2 * np.pi, (np.divide(d, 2))))

        # X2: sig^-1/2
        x2 = np.power(np.linalg.det(sigma_k), (-0.5))

        # X3: Tricky vectorization part, exp(-1/2*invsig*x-mu*x-mu.T)
        x3 = np.dot(invsig_k, x_sub_mus.T).T
        x3 = np.sum(x_sub_mus * x3, axis=1)
        x3 = np.exp(np.multiply((-0.5), x3))

        # Combined formula
        r = x1 * x2 * x3

        # Multiply by phi
        r = PI[kk] * r

        responsibility[kk, :] = r

    llk = np.sum(np.log10(np.sum(responsibility, axis=0)))
    return llk

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        MU, SIGMA, PI = initialize_parameters(X, k)
        SIGMA = compute_sigma(X, MU)
    else:
        MU, SIGMA, PI = initial_values

    # initial_likelihood = likelihood(X, PI, MU, SIGMA, k)
    prev_likelihood = 0
    new_likelihood = None
    count = 0
    count_max = 10

    while True:

        # e step
        r = E_step(X, MU, SIGMA, PI, k)

        # m step
        MU, SIGMA, PI = M_step(X, r, k)

        # likelihood
        if new_likelihood is not None:
            prev_likelihood = new_likelihood
        new_likelihood = likelihood(X, PI, MU, SIGMA, k)
        # print("new likelihood: ", new_likelihood)

        # convergence
        count, converged = convergence_function(prev_likelihood, new_likelihood, count, count_max)

        if converged:
            # use new parameters if score low
            # if new_likelihood-initial_likelihood < 90000:
            #     print("new means, improvement score: ", new_likelihood-initial_likelihood)
            #     MU, SIGMA, PI = initialize_parameters(X, k)
            #     continue
            # else:
            return MU, SIGMA, PI, r

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood
    (maximum responsibility value).

    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    return:
    clusters = numpy.ndarray[int] - m x 1
    """
    return np.argmax(r, axis=0)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def segment(X, MU, k, r):
    """
    Segment the X matrix into k components.
    Returns a matrix where each data point is
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood
    component mean. (the shape is still mxn, not
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """
    clusters = cluster(r)
    new_X = np.empty_like(X)
    new_X = np.append(new_X, clusters[:, np.newaxis], axis=1)  # Attach best clusters to new_X
    for i in range(k):
        curr_mean = MU[i]
        new_X[new_X[:, -1] == i, 0:-1] = curr_mean

    return new_X[:, 0:-1]

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    segment_arr = []
    for i in range(iters):
        print("Training iteration {}".format(i))
        MU, SIGMA, PI, r = train_model(X, k, convergence_function=default_convergence)
        likey = likelihood(X, PI, MU, SIGMA, k)
        seg = segment(X, MU, k, r)
        segment_arr.append((likey, seg))

    best_seg = max(segment_arr, key=lambda item: item[0])

    return best_seg

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    """
    MU, SIGMA, PI = initialize_parameters(X, k)
    MU, new_clusters = k_means_step(X, k, MU)

    return MU, SIGMA, PI
    #
    # current_likelihood = None
    # best_params = None
    # range_max = 100
    # for t in range(1, range_max):
    #     if t%100 == 0:
    #         print("Round ", t, "  Current likelihood: ", current_likelihood)
    #     T = range_max / t
    #     if T == 0:
    #         print("T==0, best likelihood found: ", current_likelihood)
    #         continue
    #
    #     # Randomly selected successor
    #     rng = np.random.default_rng()
    #     numbers = rng.choice(X.shape[0], size=k, replace=False)
    #     initial_means = X[numbers]
    #     r = E_step(X, initial_means, SIGMA, PI, k)
    #     MU, SIGMA, PI = M_step(X, r, k)
    #     next_likelihood = likelihood(X, PI, MU, SIGMA, k)
    #
    #     # Simulated annealing
    #     if current_likelihood is None:
    #         current_likelihood = next_likelihood
    #     delt_E = next_likelihood-current_likelihood
    #     if delt_E > 0:
    #         current_likelihood = next_likelihood
    #         best_params = MU, SIGMA, PI
    #     else:
    #         chance_next = np.exp(2.71828182845**(delt_E/T))/100
    #         if chance_next == np.nan:
    #             chance_next = 0
    #         if np.random.choice([True, False], p=[chance_next, 1-chance_next]):
    #             current_likelihood = next_likelihood
    #             best_params = MU, SIGMA, PI
    #
    # MU, SIGMA, PI = best_params
    # #------------------------------------------------
    #
    # return MU, SIGMA, PI


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """
    lower = 0.97
    upper = 1.03
    pMU, pSIGMA, pPI = previous_variables
    nMU, nSIGMA, nPI = new_variables

    # Mean check
    meanThresh = True
    nMU = nMU.flatten()
    pMU = pMU.flatten()
    for i in range(len(nPI)):
        if not (abs(pMU[i]) * lower < abs(nMU[i]) < abs(pMU[i]) * upper):
            meanThresh = False
            break

    # Sigma check
    sigThresh = True
    nSIGMA = nSIGMA.flatten()
    pSIGMA = pSIGMA.flatten()
    for i in range(len(nSIGMA)):
        if not (abs(pSIGMA[i]) * lower < abs(nSIGMA[i]) < abs(pSIGMA[i]) * upper):
            sigThresh = False
            break

    # Pi check
    piThresh = True
    for i in range(len(nPI)):
        if not (abs(pPI[i]) * lower < abs(nPI[i]) < abs(pPI[i]) * upper):
            piThresh = False
            break

    increase_convergence_ctr = meanThresh and sigThresh and piThresh

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap

def train_model_improved(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction
    implemented above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        MU, SIGMA, PI = improved_initialization(X, k)
        SIGMA = compute_sigma(X, MU)
    else:
        #MU, SIGMA, PI = initial_values
        MU, SIGMA, PI = improved_initialization(X, k)

    new_vars = None
    count = 0
    count_max = 10

    convergences_lk = []
    convergences_params = []
    convergences_ct = 0
    while True:
        prev_vars = MU, SIGMA, PI
        r = E_step(X, MU, SIGMA, PI, k)
        MU, SIGMA, PI = M_step(X, r, k)
        new_vars = MU, SIGMA, PI
        count, converged = convergence_function(prev_vars, new_vars, count, count_max)

        # use new parameters if score low
        # if new_likelihood-initial_likelihood < 90000:
        #     print("new means, improvement score: ", new_likelihood-initial_likelihood)
        #     MU, SIGMA, PI = initialize_parameters(X, k)
        #     continue
        # else:

        if converged:
            print("converge ", convergences_ct)
            convergences_lk.append(likelihood(X, PI, MU, SIGMA, k))
            convergences_params.append((MU, SIGMA, PI, r))
            convergences_ct += 1
            if convergences_ct > 10:
                return convergences_params[convergences_lk.index(max(convergences_lk))]

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
# Unittest below will check both of the functions at the same time.
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def count_params(MU, X, k):
    dim = MU.shape[1]
    n_observations = X.shape[0]
    mu_params = dim
    # cov_params = (dim * (dim + 1) / 2)  # -1 for middle???
    cov_params = 2*dim
    mu_params = 1
    pi_params = 1
    k_params = k * (mu_params + cov_params + pi_params)
    return k_params, n_observations

def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    return:
    bayes_info_criterion = int
    """
    # params_k = k
    # params_P = 1
    # params_MU = MU.shape[0]
    # params_SIG = int((SIGMA.shape[1]*SIGMA.shape[2]) / 2)
    # k_params = (params_k * params_P * params_MU * params_SIG)
    #
    # n = X.shape[0]
    # lik = likelihood(X, PI, MU, SIGMA, k)
    # bic = np.log10(n) * k_params - 2 * lik
    # print(bic)

    k_params, n = count_params(MU, X, k)
    lik = likelihood(X, PI, MU, SIGMA, k)
    bic = np.log10(n) * k_params - (2 * lik)
    print(bic)
    return bic

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC
    and maximum likelihood with respect
    to image_matrix and comp_means.

    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    print("Shape image")
    print(image_matrix.shape)

    print("Printing list of means")
    for z in comp_means:
        print(z)

    bic_scores = []
    like_scores = []
    k_vals = []
    mean_vals = []

    for m in range(0, len(comp_means)):
        curr_means = comp_means[m]

        # derive parameters
        curr_k = curr_means.shape[0]

        MU, SIGMA, PI, r = train_model_improved(image_matrix, curr_k, convergence_function=new_convergence_function)

        # PI = np.ones([curr_k])
        # PI = PI / curr_k
        # SIGMA = compute_sigma(image_matrix, curr_means)

        # keep track of bic and likelihood
        b_i_c = bayes_info_criterion(image_matrix, PI, curr_means, SIGMA, curr_k)
        bic_scores.append(b_i_c)

        likey = likelihood(image_matrix, PI, MU, SIGMA, curr_k)
        like_scores.append(likey)

        k_vals.append(curr_k)
        mean_vals.append(curr_means)

    # retrieve best bic
    bic_i = bic_scores.index(min(bic_scores))
    #bic_param_count, bic_observations = count_params(mean_vals[bic_i], image_matrix, k_vals[bic_i])
    bic_k = k_vals[bic_i]

    # retrieve best likelihood
    like_i = like_scores.index(max(like_scores))
    #likelihood_param_count, likelihood_observations = count_params(mean_vals[like_i], image_matrix, k_vals[like_i])
    like_k = k_vals[like_i]

    # print("Returning (bic param #, likelihood param): ", bic_param_count, likelihood_param_count)
    # return bic_param_count, likelihood_param_count
    print("Returning (bic k, like k): ", bic_k, like_k)
    return bic_k, like_k


def return_your_name():
    return "dward45"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.
    returns:
    dists = numpy array of float
    """
    return np.sqrt(np.sum(np.power(points_array[:, None] - means_array, 2), axis=2))


# There are no local test for thus question, fill free to create them yourself.
# Feel free to play with it in a separate python file, and then just copy over
# your implementation before the submission.