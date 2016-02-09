def make_folds(X, y, target_size, method='random'):
    n_Y = y.shape[0]
    n_folds = int(n_Y/target_size) + int(target_size>n_Y)

    if method == 'random':
        fold_assignment = np.random.permutation(n_Y)%n_folds
    elif method == 'cluster':
        # Thanks scikit
        print('Clustering [sklearn.cluster] inputs')
        clusterer = skcluster.MiniBatchKMeans(n_clusters=n_folds, batch_size=1000)
        fold_assignment = clusterer.fit_predict(X)
    elif method == 'rcluster':
        print('Clustering [sklearn.cluster] inputs')
        clusters = skcluster.MiniBatchKMeans(n_clusters=n_folds,
                                batch_size=1000, compute_labels=True).fit(X)
        Xcluster = clusters.cluster_centers_
        print('Interpolating probability')
        n_X = X.shape[0]
        assign_prob = np.zeros((n_folds, n_X))
        tris = Delaunay(Xcluster)
        base_labels = clusters.labels_
        for i in range(n_folds):
            indicator = np.zeros(n_folds)
            indicator[i] = 1.
            row = interp.LinearNDInterpolator(tris, indicator,
                                                         fill_value=-1)(X)
            row[row<0] = base_labels[row<0] == i
            assign_prob[i] = row

        # now use these as selection probabilities
        assign_prob = np.cumsum(assign_prob, axis=0)

        rvec = np.random.random(n_X)
        fold_assignment = np.sum(rvec[np.newaxis, :] <assign_prob, axis=0)

        # veryfy fold assignment?
        # pl.scatter(X[:, 0], X[:, 1], c=fold_assignment)
        # pl.show()
        # exit()

    else:
        raise NameError('Unrecognised fold method:'+method)

    fold_inds = np.unique(fold_assignment)
    folds = Folds(n_folds, [], [], [])  # might contain lists in the multitask case
    where = lambda y, v:y[np.where(v)[0]]
    for f in fold_inds:
        folds.X.append(where(X, fold_assignment==f))
        folds.Y.append(where(y, fold_assignment==f))
        folds.flat_y.append(where(y, fold_assignment==f))

    return folds




def chol_up(L, Sn, Snn, Snn_noise_std_vec):
    # Incremental cholesky update
    Ln = la.solve_triangular(L, Sn, lower=True).T
    On = np.zeros(Ln.shape).T
    noise = np.diag(Snn_noise_std_vec ** 2)
    Lnn = linalg.jitchol(Snn+noise - Ln.dot(Ln.T))
    top = np.concatenate((L, On), axis=1)
    bottom = np.concatenate((Ln, Lnn), axis=1)
    return np.concatenate((top, bottom), axis=0)


def chol_up_insert(L, V12, V23, V22, Snn_noise_std_vec, insertionID):

    R = L.T
    N = R.shape[0]
    n = V22.shape[0]
    noise = np.diag(Snn_noise_std_vec ** 2)
    R11 = R[:insertionID, :insertionID]
    R33 = R[insertionID:, insertionID:]
    S11 = R11
    S12 = la.solve_triangular(R11.T, V12, lower=True)
    S13 = R[:insertionID, insertionID:]
    S22 = linalg.jitchol(V22+noise - S12.T.dot(S12)).T
    if V23.shape[1] != 0:  # The data is being inserted between columns
        S23 = la.solve_triangular(S22.T, (V23-S12.T.dot(S13)), lower=True)
        S33 = linalg.jitchol(R33.T.dot(R33)-S23.T.dot(S23)).T
    else:  #the data is being appended at the end of the matrix
        S23 = np.zeros((n, 0))
        S33 = np.zeros((0, 0))
    On1 = np.zeros((n, insertionID))
    On2 = np.zeros((N-insertionID, insertionID))
    On3 = np.zeros((N-insertionID, n))

    top = np.concatenate((S11, S12, S13), axis=1)
    middle = np.concatenate((On1, S22, S23), axis=1)
    bottom = np.concatenate((On2, On3, S33), axis=1)
    return np.concatenate((top, middle, bottom), axis=0).T

def chol_down(L, remIDList):
    # This works but it might potentially be slower than the naive approach of
    # recomputing the cholesky decomposition from scratch.
    # The jitchol line can apparently be replaces with a chol that exploits the
    # structure of the problem according to Osbourne's Thesis (as
    # cholupdate does).
    remIDList = np.sort(remIDList)
    for i in range(len(remIDList)):
        remID = remIDList[i]
        S = L.T
        n = S.shape[0]
        On = np.zeros((n-(remID+1), remID))
        # Incremental cholesky downdate
        top = np.concatenate((S[:remID, :remID], S[:(remID), (remID+1):]), axis=1)
        S23 = S[remID, (remID+1):][np.newaxis, :]
        S23TS23 = S23.T.dot(S23)
        S33TS33 = S[(remID+1):, (remID+1):].T.dot(S[(remID+1):, (remID+1):])
        R33 = linalg.jitchol(S23TS23+S33TS33).T
        bottom = np.concatenate((On, R33), axis=1)
        L = np.concatenate((top, bottom), axis=0).T
        remIDList -= 1
    return L


def add_data(newX, newY, regressor, query=None, insertionID=None):
    assert(isinstance(regressor, dtypes.RegressionParams))
    assert(not query or isinstance(query, dtypes.QueryParams))
    assert(len(newX.shape) == 2)
    assert(len(newY.shape) == 1)

    if not(insertionID):  #No insterionID provide. Append data to the end.
        # Compute the new rows and columns of the covariance matrix
        Kxn = regressor.kernel(regressor.X, newX)
        Knn = regressor.kernel(newX, newX)
        nn_noise_std = predict.noise_vector(newX, regressor.noise_std)
        # Update the regression opt_config_copys
        regressor.X = np.vstack((regressor.X, newX))
        regressor.y = np.hstack((regressor.y, newY))
        regressor.L = chol_up(regressor.L, Kxn, Knn,
                              nn_noise_std)
        # sadly, this is still expensive. However osborne's thesis appendix B can
        # be used to speed up this step too. Maybe by a factor of 2.
        regressor.alpha = predict.alpha(regressor.y, regressor.L)

        # Optionally update the query
        if query is not None:
            Kxsn = regressor.kernel(newX, query.Xs)
            query.K_xxs = np.vstack((query.K_xxs, Kxsn))
    else:
        # Compute the new rows and columns of the covariance matrix
        Kx1n = regressor.kernel(regressor.X[:insertionID, :], newX)
        Knx2 = regressor.kernel(newX, regressor.X[insertionID:, :])
        Knn = regressor.kernel(newX, newX)
        nn_noise_std = predict.noise_vector(newX, regressor.noise_std)
        regressor.X = np.vstack((regressor.X[:insertionID, :], newX,
                                 regressor.X[insertionID:, :]))
        regressor.y = np.hstack((regressor.y[:insertionID], newY,
                                 regressor.y[insertionID:]))
        regressor.L = chol_up_insert(regressor.L, Kx1n, Knx2, Knn,
                              nn_noise_std, insertionID)
        # sadly, this is still expensive. However osborne's thesis appendix B can
        # be used to speed up this step too. Maybe by a factor of 2.
        regressor.alpha = predict.alpha(regressor.y, regressor.L)

        if query is not None:
            Kxsn = regressor.kernel(newX, query.Xs)
            query.K_xxs = np.vstack((query.K_xxs[:insertionID, :], Kxsn,
                                     query.K_xxs[insertionID:, :]))


def remove_data(regressor, remID, query=None):
    assert(isinstance(regressor, dtypes.RegressionParams))
    assert(not query or isinstance(query, dtypes.QueryParams))


    regressor.X = np.delete(regressor.X, remID, axis=0)
    regressor.y = np.delete(regressor.y, remID, axis=0)
    # regressor.L = chol_down(regressor.L, remID)


    noise_vector = predict.noise_vector(regressor.X, regressor.noise_std)
    regressor.L = linalg.cholesky(regressor.X, regressor.kernel, noise_vector)
    regressor.alpha = predict.alpha(regressor.y, regressor.L)

    # Optionally update the query
    if query is not None:
        query.K_xxs = np.delete(query.K_xxs, remID, axis=0)




def learn_folds(folds, cov_fn, optParams, optCrition='logMarg', verbose=False):
    # Same as learn, but using multiple folds jointly
    # todo: distribute computation!
    def criterion(sigma, noise):
        k = lambda x1, x2: cov_fn(x1, x2, sigma)
        val = 0
        for f in range(folds.n_folds):
            Xf = folds.X[f]
            Yf = folds.flat_y[f]
            Xf_noise = predict.noise_vector(Xf, noise)
            Lf = linalg.cholesky(Xf, k, Xf_noise)
            af = predict.alpha(Yf, Lf)
            if optCrition == 'logMarg':
                val += negative_log_marginal_likelihood(Yf, Lf, af)
            elif optCrition == 'crossVal':
                val += negative_log_prob_cross_val(Yf, Lf, af)
        if verbose is True:
            print('['+str(val)+']  ', sigma, noise)
        return val

    sigma, noise, optval = optimise_hypers(criterion, optParams)

    if verbose:
        print('[', optval, ']:', sigma, noise)
    return sigma, noise
