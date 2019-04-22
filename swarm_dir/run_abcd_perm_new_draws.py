import statsmodels.formula.api as smf
import statsmodels as sm
from collections import namedtuple
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.externals import joblib

import patsy
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
class ResidTransform(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self, confound):
        self.confound = confound
        
    def fit(self, x, y = None):
        self.lm = LinearRegression().fit(self.confound,x)
        return self
    
    def transform(self, x, confound=None):
        if confound is None:
            confound = self.confound
        return x - self.lm.predict(confound)
robjects.r('library(sva)')
robjects.r('library(BiocParallel)')
robjects.r('''modifiedComBat <- function (dat, batch, mod = NULL, par.prior = FALSE, prior.plots = FALSE, 
           mean.only = FALSE, ref.batch = NULL, BPPARAM = bpparam("SerialParam"), dat2 = NULL, batch2 = NULL, mod2 = NULL) 
{
  if (mean.only == TRUE) {
    message("Using the 'mean only' version of ComBat")
  }
  if (length(dim(batch)) > 1) {
    stop("This version of ComBat only allows one batch variable")
  }
  if (is.null(batch2)) {
    batch <- as.factor(batch)
  }
  else {
    n1 <- length(batch)
    allbatch <- as.factor(c(batch, batch2))
    batch <- allbatch[1:n1]
    batch2 <- allbatch[(n1 + 1):length(allbatch)]
  }
  batchmod <- model.matrix(~-1 + batch)
  if (!is.null(dat2)) batchmod2 <- model.matrix(~-1 + batch2)
  if (!is.null(ref.batch)) {
    if (!(ref.batch %in% levels(batch))) {
      stop("reference level ref.batch is not one of the levels of the batch variable")
    }
    cat("Using batch =", ref.batch, "as a reference batch (this batch won't change)\n")
    ref <- which(levels(as.factor(batch)) == ref.batch)
    batchmod[, ref] <- 1
    if (!is.null(dat2)) {
      ref2 <- which(levels(as.factor(batch2)) == ref.batch)
      batchmod2[, ref2] <- 1
    }
  }
  else {
    ref <- NULL
  }
  message("Found", nlevels(batch), "batches")
  n.batch <- nlevels(batch)
  batches <- list()
  for (i in 1:n.batch) {
    batches[[i]] <- which(batch == levels(batch)[i])
  }
  if (!is.null(dat2)) {
    batches2 <- list()
    for (i in 1:n.batch) {
      batches2[[i]] <- which(batch2 == levels(batch)[i])
    }
  }
  n.batches <- sapply(batches, length)
  if (any(n.batches == 1)) {
    mean.only = TRUE
    message("Note: one batch has only one sample, setting mean.only=TRUE")
  }
  n.array <- sum(n.batches)
  design <- cbind(batchmod, mod)
  check <- apply(design, 2, function(x) all(x == 1))
  if (!is.null(ref)) {
    check[ref] <- FALSE
  }
  design <- as.matrix(design[, !check])
  if (!is.null(dat2)) {
    n.batches2 <- sapply(batches2, length)
    n.array2 <- sum(n.batches2)
    design2 <- cbind(batchmod2, mod2)
    check2 <- apply(design2, 2, function(x) all(x == 1))
    if (!is.null(ref)) {
      check2[ref] <- FALSE
    } 
    design2 <- as.matrix(design2[, !check2])
  }
  message("Adjusting for", ncol(design) - ncol(batchmod), "covariate(s) or covariate level(s)")
  if (qr(design)$rank < ncol(design)) {
    if (ncol(design) == (n.batch + 1)) {
      stop("The covariate is confounded with batch! Remove the covariate and rerun ComBat")
    }
    if (ncol(design) > (n.batch + 1)) {
      if ((qr(design[, -c(1:n.batch)])$rank < ncol(design[, 
                                                          -c(1:n.batch)]))) {
        stop("The covariates are confounded! Please remove one or more of the covariates so the design is not confounded")
      }
      else {
        stop("At least one covariate is confounded with batch! Please remove confounded covariates and rerun ComBat")
      }
    }
  }
  NAs <- any(is.na(dat))
  if (NAs) {
    message(c("Found", sum(is.na(dat)), "Missing Data Values"), 
            sep = " ")
  }
  cat("Standardizing Data across genes\n")
  
  if (!NAs) {
    B.hat <- solve(crossprod(design), tcrossprod(t(design), 
                                                 as.matrix(dat)))
  }
    else {
      B.hat <- apply(dat, 1, Beta.NA, design)
    }
  if (!is.null(ref.batch)) {
    grand.mean <- t(B.hat[ref, ])
  }
    else {
      grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch, 
                                                       ])
    }
  if (!NAs) {
    if (!is.null(ref.batch)) {
      ref.dat <- dat[, batches[[ref]]]
      var.pooled <- ((ref.dat - t(design[batches[[ref]], 
                                         ] %*% B.hat))^2) %*% rep(1/n.batches[ref], n.batches[ref])
    }
    else {
      var.pooled <- ((dat - t(design %*% B.hat))^2) %*% 
        rep(1/n.array, n.array)
    }
  }
    else {
      if (!is.null(ref.batch)) {
        ref.dat <- dat[, batches[[ref]]]
        var.pooled <- rowVars(ref.dat - t(design[batches[[ref]], 
                                                 ] %*% B.hat), na.rm = TRUE)
      }
      else {
        var.pooled <- rowVars(dat - t(design %*% B.hat), 
                              na.rm = TRUE)
      }
    }
  stand.mean <- t(grand.mean) %*% t(rep(1, n.array))
  if (!is.null(dat2)) stand.mean2 <- t(grand.mean) %*% t(rep(1, n.array2))
  if (!is.null(design)) {
    tmp <- design
    tmp[, c(1:n.batch)] <- 0
    stand.mean <- stand.mean + t(tmp %*% B.hat)
    if (!is.null(dat2)) {
      tmp2 <- design2
      tmp2[, c(1:n.batch)] <- 0
      stand.mean2 <- stand.mean2 + t(tmp2 %*% B.hat)
    }
  }
  s.data <- (dat - stand.mean)/(sqrt(var.pooled) %*% t(rep(1, 
                                                           n.array)))
  if (!is.null(dat2)) {
    s.data2 <- (dat2 - stand.mean2)/(sqrt(var.pooled) %*% t(rep(1, 
                                                             n.array2)))
  }
  message("Fitting L/S model and finding priors")
  batch.design <- design[, 1:n.batch]
  if (!is.null(dat2)) batch.design2 <- design2[, 1:n.batch]
  if (!NAs) {
    gamma.hat <- solve(crossprod(batch.design), tcrossprod(t(batch.design), 
                                                           as.matrix(s.data)))
  }
    else {
      gamma.hat <- apply(s.data, 1, Beta.NA, batch.design)
    }
  delta.hat <- NULL
  for (i in batches) {
    if (mean.only == TRUE) {
      delta.hat <- rbind(delta.hat, rep(1, nrow(s.data)))
    }
    else {
      delta.hat <- rbind(delta.hat, rowVars(s.data[, i], 
                                            na.rm = TRUE))
    }
  }
  gamma.bar <- rowMeans(gamma.hat)
  t2 <- rowVars(gamma.hat)
  a.prior <- apply(delta.hat, 1, sva:::aprior)
  b.prior <- apply(delta.hat, 1, sva:::bprior)
  if (prior.plots && par.prior) {
    par(mfrow = c(2, 2))
    tmp <- density(gamma.hat[1, ])
    plot(tmp, type = "l", main = expression(paste("Density Plot of First Batch ", 
                                                  hat(gamma))))
    xx <- seq(min(tmp$x), max(tmp$x), length = 100)
    lines(xx, dnorm(xx, gamma.bar[1], sqrt(t2[1])), col = 2)
    qqnorm(gamma.hat[1, ], main = expression(paste("Normal Q-Q Plot of First Batch ", 
                                                   hat(gamma))))
    qqline(gamma.hat[1, ], col = 2)
    tmp <- density(delta.hat[1, ])
    xx <- seq(min(tmp$x), max(tmp$x), length = 100)
    tmp1 <- list(x = xx, y = sva:::dinvgamma(xx, a.prior[1], b.prior[1]))
    plot(tmp, typ = "l", ylim = c(0, max(tmp$y, tmp1$y)), 
         main = expression(paste("Density Plot of First Batch ", 
                                 hat(delta))))
    lines(tmp1, col = 2)
    invgam <- 1/qgamma(1 - ppoints(ncol(delta.hat)), a.prior[1], 
                       b.prior[1])
    qqplot(invgam, delta.hat[1, ], main = expression(paste("Inverse Gamma Q-Q Plot of First Batch ", 
                                                           hat(delta))), ylab = "Sample Quantiles", xlab = "Theoretical Quantiles")
    lines(c(0, max(invgam)), c(0, max(invgam)), col = 2)
  }
  gamma.star <- delta.star <- matrix(NA, nrow = n.batch, ncol = nrow(s.data))
  if (par.prior) {
    message("Finding parametric adjustments")
    results <- BiocParallel:::bplapply(1:n.batch, function(i) {
      if (mean.only) {
        gamma.star <- postmean(gamma.hat[i, ], gamma.bar[i], 
                               1, 1, t2[i])
        delta.star <- rep(1, nrow(s.data))
      }
      else {
        temp <- sva:::it.sol(s.data[, batches[[i]]], gamma.hat[i, 
                                                               ], delta.hat[i, ], gamma.bar[i], t2[i], a.prior[i], 
                             b.prior[i])
        gamma.star <- temp[1, ]
        delta.star <- temp[2, ]
      }
      list(gamma.star = gamma.star, delta.star = delta.star)
    }, BPPARAM = BPPARAM)
    for (i in 1:n.batch) {
      gamma.star[i, ] <- results[[i]]$gamma.star
      delta.star[i, ] <- results[[i]]$delta.star
    }
  }
    else {
      message("Finding nonparametric adjustments")
      results <- BiocParallel:::bplapply(1:n.batch, function(i) {
        if (mean.only) {
          delta.hat[i, ] = 1
        }
        temp <- sva:::int.eprior(as.matrix(s.data[, batches[[i]]]), 
                           gamma.hat[i, ], delta.hat[i, ])
        list(gamma.star = temp[1, ], delta.star = temp[2, 
                                                       ])
      }, BPPARAM = BPPARAM)
      for (i in 1:n.batch) {
        gamma.star[i, ] <- results[[i]]$gamma.star
        delta.star[i, ] <- results[[i]]$delta.star
      }
    }
  if (!is.null(ref.batch)) {
    gamma.star[ref, ] <- 0
    delta.star[ref, ] <- 1
  }
  message("Adjusting the Data\n")
  bayesdata <- s.data
  j <- 1
  for (i in batches) {
    bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i, 
                                                       ] %*% gamma.star))/(sqrt(delta.star[j, ]) %*% t(rep(1, 
                                                                                                           n.batches[j])))
    j <- j + 1
  }
  bayesdata <- (bayesdata * (sqrt(var.pooled) %*% t(rep(1, 
                                                        n.array)))) + stand.mean
  if (!is.null(ref.batch)) {
    bayesdata[, batches[[ref]]] <- dat[, batches[[ref]]]
  }
  if (!is.null(dat2)) {
    bayesdata2 <- s.data2
    j <- 1
    for (i in batches2) {
      bayesdata2[, i] <- (bayesdata2[, i] - t(batch.design2[i, 
                                                         ] %*% gamma.star))/(sqrt(delta.star[j, ]) %*% t(rep(1, 
                                                                                                             n.batches2[j])))
      j <- j + 1
    }
    bayesdata2 <- (bayesdata2 * (sqrt(var.pooled) %*% t(rep(1, 
                                                          n.array2)))) + stand.mean2
    if (!is.null(ref.batch)) {
      bayesdata2[, batches2[[ref]]] <- dat2[, batches2[[ref]]]
    }
    return(list(corrected = bayesdata, alpha = grand.mean, beta.hat = B.hat, gamma.star = gamma.star, delta.star = delta.star,
                corrected2 = bayesdata2))
  }
  return(list(corrected = bayesdata, alpha = grand.mean, beta.hat = B.hat, gamma.star = gamma.star, delta.star = delta.star))
}
''')
combat = robjects.r('modifiedComBat')


def run_combat(feats, meta, model="~interview_age + gender + ehi_ss_score",
              feats_test=None, meta_test=None, model_test=None):
    model_matrix = patsy.dmatrix(model, meta)
    fmat = np.array(feats).T
    rbatch = robjects.IntVector(pd.Categorical(meta.unique_scanner).codes)
    
    if (meta_test is not None) and (feats_test is not None):
        if model_test is None:
            model_test = model
        model_matrix_test = patsy.dmatrix(model_test, meta_test)
        fmat_test = np.array(feats_test).T
        rbatch_test = robjects.IntVector(pd.Categorical(meta_test.unique_scanner).codes)
        combat_result = combat(dat=fmat, batch=rbatch, mod=model_matrix,
                               dat2=fmat_test, batch2=rbatch_test, mod2=model_matrix_test)
    else:
        combat_result = combat(dat = fmat, batch = rbatch, mod = model_matrix)
    combat_result = [np.array(cr) for cr in combat_result]
    return combat_result


def bal_samp(df, strata, balance, order, keys, n_splits=5, n_draws=100):
    """Balanced sampling across strata
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe from which you want to sample
    strata: str or list of str
        Name(s) of the column or columns that define the groups
        from which you want a balanced sample
    balance: str or list of str
        Name(s) of the columns or columns containing the factors
        you want to evenly sample across strata
    order: str or list of str
        Name(s) of the column whose distribution you want to preserve
    keys: list of str
        Name(s) of the column(s) that you will use to match
        the output back to your original column
    n_splits: int
        Number of cross validation folds you want to create per draw
    n_draws: int
        Number of balanced samples of your dataset you want to create
    
    Returns
    -------
    draws_df: pandas.DataFrame
        Dataframe with number of rows about equal to number of rows in df
        and number of columns equal to n_draws + len(keys) + len(strata) + len(balance) + len(order).
        Contains the crossfold labels for balanced sampling across the strata you've defined.
    """
    # create dict of minimum count in each strata of each combination of balance factor
    bal_dict = (df.groupby(strata + balance)[[keys[0]]]
                  .nunique()
                  .groupby(balance)
                  .min()
                  .to_dict('index'))
    bal_dict = {k:v[keys[0]] for k,v in bal_dict.items()}
    
    # Appologies for the disgusting nested loops
    # For each draw, at each strata level, for each unique combin
    draws_df = []
    # For each draw
    for nn in range(n_draws):
        strat_df = []
        # From each strata group
        for x, gbdf in df.groupby(strata):
            cvs_df = []
            # from each unique combination of balance values
            for bal_vals, num in bal_dict.items():
                # create an index selecting the rows at those balance values
                ind = np.ones((len(gbdf))).astype(bool)
                for bcol,bv in zip(balance,bal_vals):
                    ind = np.logical_and(ind, gbdf[bcol] == bv)
                # draw a random sample of the group members 
                # that meet the balance criteria
                # and sort them by the order values
                bal_df = gbdf[ind].sample(n=num).sort_values(order).loc[:,keys]
                # create a list of the cross validation values long enough to match
                cv_inds = list(np.arange(n_splits))*((len(bal_df)//n_splits)+1)
                bal_df['draw_%d'%nn] = cv_inds[:len(bal_df)]
                # and append them to a list
                cvs_df.append(bal_df)
            # combine these lists to get all the rows for that strata
            # and append them to create a list of selected rows from all the strata
            strat_df.append(pd.concat(cvs_df).loc[:,['draw_%d'%nn]])
        # pull these all together to create the draws dataframe
        draws_df.append(pd.concat(strat_df))
    draws_df = pd.concat(draws_df, axis=1)
    # Merge back in the indicator variables
    draws_df = (df.loc[:,
                      keys 
                      + strata 
                      + balance 
                      + order]
                  .merge(draws_df,
                         right_index=True,
                         left_index=True,
                         how='left')
               )
    # make sure the shape is still ok
    assert draws_df.shape[0] == df.shape[0]
    assert draws_df.shape[1] == (n_draws
                                 + len(keys)
                                 + len(strata)
                                 + len(balance)
                                 + len(order))
    return draws_df


ModPermRes = namedtuple("ModPermRes",['pn','fn','name', 'clf', 'ho_score','cfn'])
VarPermRes = namedtuple("VarPermRes", ["pn", "metric", "int_r2", "agh_r2", "aghs_r2", "aghss_r2", "aghsss_r2"])


def fit_model(X_r, Y_r, X_e, Y_e, pn, fn, name, mapper):
    clf = Pipeline([('preprocessing', mapper),
                    ('clf', LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 10000,
                                               penalty='l2',
                                               C=1,
                                               fit_intercept=True))])

    clf.fit(X_r, Y_r)

    res = ModPermRes(pn,
                     fn,
                     name,
                     clf.named_steps['clf'],
                     clf.score(X_e, Y_e),
                     confusion_matrix(Y_e, clf.predict(X_e)))
    return res


def get_splits(draws_df, n_draws=None, dn=None):
    cv_splits = []
    if dn is None and n_draws is None:
        raise ValueError("one of n_draws or dn must be set")
    elif dn is None:
        for dn in range(n_draws):
            for fn in range(int(draws_df['draw_%d'%dn].max())+1):
                tmp_df = draws_df.loc[pd.notnull(draws_df['draw_%d'%dn]), 'draw_%d'%dn]
                cv_splits.append((tmp_df[(tmp_df != fn)].index.values, tmp_df[tmp_df == fn].index.values))
    else:
        for fn in range(int(draws_df['draw_%d'%dn].max())+1):
                tmp_df = draws_df.loc[pd.notnull(draws_df['draw_%d'%dn]), 'draw_%d'%dn]
                cv_splits.append((tmp_df[(tmp_df != fn)].index.values, tmp_df[tmp_df == fn].index.values))
    return cv_splits


def per_fold(X_r, Y_r, X_e, Y_e, pn, fn, name, metric_cols):
    """Per fold function that unwraps the inner loop
       so each permutation can run all its fits in parallel."""
    
    mapper = DataFrameMapper([([nv],preprocessing.StandardScaler()) for nv in metric_cols])
    
    # Fit normal on train, score on test
    if 'normal' in name:
        print("normal", name)
        return fit_model(X_r, Y_r, X_e, Y_e, pn, fn, name, mapper)
    
    # Learn age on training, apply tp test, fit on x_r_rsd, score on x_e_rsd
    elif 'age_rsd' in name:
        print("age_rsd", name)
        X_r_rsd = X_r.copy(deep=True)
        X_e_rsd = X_e.copy(deep=True)
        for col in metric_cols:
            rt = ResidTransform(X_r_rsd.interview_age.values.reshape((-1,1)))
            X_r_rsd.loc[:,col] = rt.fit_transform(X_r_rsd.loc[:,col].values)
            X_e_rsd.loc[:,col] = rt.transform(X_e_rsd.loc[:,col].values, X_e_rsd.interview_age.values.reshape((-1,1)))
        return fit_model(X_r_rsd, Y_r, X_e_rsd, Y_e, pn, fn, name, mapper)

    elif 'cbagersd' in name:
        print("cbagersd", name)
        X_r_rsd = X_r.copy(deep=True)
        X_e_rsd = X_e.copy(deep=True)
        for col in metric_cols:
            rt = ResidTransform(X_r_rsd.interview_age.values.reshape((-1,1)))
            X_r_rsd.loc[:,col] = rt.fit_transform(X_r_rsd.loc[:,col].values)
            X_e_rsd.loc[:,col] = rt.transform(X_e_rsd.loc[:,col].values, X_e_rsd.interview_age.values.reshape((-1,1)))
        cb_meta_cols = ['unique_scanner', 'interview_age', 'gender', 'ehi_ss_score']
        X_r_cb = X_r_rsd
        X_e_cb = X_e_rsd
        # Fit combat seperately on training and test splits
        combat_res_r = run_combat(X_r_cb.loc[:, metric_cols], X_r_cb.loc[:, cb_meta_cols])
        X_r_cb.loc[:, metric_cols] = combat_res_r[0].T
        combat_res_e = run_combat(X_e_cb.loc[:, metric_cols], X_e_cb.loc[:, cb_meta_cols])
        X_e_cb.loc[:, metric_cols] = combat_res_e[0].T
        return fit_model(X_r_cb, Y_r, X_e_cb, Y_e, pn, fn, name, mapper)
    
    # Learn combat on training, apply to test, fit on x_r_cb, score on x_e_cb
    elif 'combat' in name:
        print("combat", name)
        cb_meta_cols = ['unique_scanner', 'interview_age', 'gender', 'ehi_ss_score']
        X_r_cb = X_r.copy(deep=True)
        X_e_cb = X_e.copy(deep=True)
        combat_res_r = run_combat(X_r_cb.loc[:, metric_cols], X_r_cb.loc[:, cb_meta_cols])
        X_r_cb.loc[:, metric_cols] = combat_res_r[0].T
        combat_res_e = run_combat(X_e_cb.loc[:, metric_cols], X_e_cb.loc[:, cb_meta_cols])
        X_e_cb.loc[:, metric_cols] = combat_res_e[0].T
        return (fit_model(X_r_cb, Y_r, X_e_cb, Y_e, pn, fn, name, mapper))


def run_variance_metric_perm(pn, perm, metric, df):
    # Don't forget to z-score df before this step
    df['perm_metric'] = df.copy().loc[perm,metric].values
    res = VarPermRes(pn,
                  metric,
                  smf.ols('perm_metric ~ 1 ', data = df).fit().rsquared,
                  smf.ols('perm_metric ~ interview_age + gender + ehi_ss_score ', data = df).fit().rsquared,
                  smf.ols('perm_metric ~ interview_age + gender + ehi_ss_score + scanner_manufacturer_pd', data = df).fit().rsquared,
                  smf.ols('perm_metric ~ interview_age + gender + ehi_ss_score + scanner_manufacturer_pd + scanner_type_pd', data = df).fit().rsquared,
                  smf.ols('perm_metric ~ interview_age + gender + ehi_ss_score + scanner_manufacturer_pd + scanner_type_pd + unique_scanner', data = df).fit().rsquared)
    return res


def run_variance_perm(pn, perm, df, metrics):
    return [run_variance_metric_perm(pn, perm, metric, df) for metric in metrics]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('pn', type=int, help="Integer specifying which permutation to run")
    parser.add_argument('perm_path', type=str, help="Path to pickle specifying permuted indices")
    parser.add_argument('ymapper_path', type=str, help="Path to pickled ymapper")
    parser.add_argument('raw_df_path', type=str, help="Path to pickled raw dataframe")
    #parser.add_argument('draws_df_path', type=str, help="Path to pickled draw dataframe")

    parser.add_argument('out_path', type=str, help="Path dump pickle")
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--n_draws', type=int, action='store')
    parser.add_argument('--dn', type=int, action='store')
    parser.add_argument('--do_ab', action='store_true')
    #parser.add_argument('--ab_draws_df_path', type=str, action='store',
    #                    help="Path to pickled draw dataframe for age balanced data, optional, if not present, age balanced analysis isn't run.")

    args = parser.parse_args()
    print(args, flush=True)
    # Load data
    pn = args.pn
    with open(args.perm_path, 'rb') as h:
        perms = pickle.load(h) 
    with open(args.ymapper_path, 'rb') as h:
        ymapper = pickle.load(h)
    raw_df = pd.read_pickle(args.raw_df_path)
    #draws_df = pd.read_pickle(args.draws_df_path)
    do_ab = args.do_ab
    #if args.ab_draws_df_path:
    #    #ab_draws_df = pd.read_pickle(args.ab_draws_df_path)
    #    do_ab = True
        
    out_path = args.out_path
    n_jobs = args.n_jobs
    n_draws = args.n_draws
    dn = args.dn

    # Pick out meta columns and metric columns
    base_meta_cols = ['contrast', 'fmri_beta_gparc_numtrs', 'fmri_beta_gparc_tr', 'lmt_run',
                      'mid_beta_seg_dof', 'task', 'collection_id', 'dataset_id', 'subjectkey',
                      'src_subject_id', 'interview_date', 'interview_age',
                      'gender', 'event_name', 'visit', 'rsfm_tr', 'eventname',
                      'rsfm_nreps', 'rsfm_numtrs', 'pipeline_version',  'scanner_manufacturer_pd',
                      'scanner_type_pd', 'mri_info_deviceserialnumber', 'magnetic_field_strength',
                      'procdate', 'collection_title', 'promoted_subjectkey', 'study_cohort_name',
                      'ehi_ss_score', '_merge', 'qc_ok', 'age_3mos', 'abcd_betnet02_id', 'fsqc_qc',
                      'rsfmri_cor_network.gordon_visitid',
                      'mrirscor02_id',  'site_id_l', 'mri_info_manufacturer',
                      'mri_info_manufacturersmn', 'mri_info_deviceserialnumber',
                      'mri_info_magneticfieldstrength', 'mri_info_softwareversion',
                      'unique_scanner', 'tbl_id', 'tbl_visitid', 
                     'modality', 'metric', 'source_file', 'tbl_id_y', 'source_file_y', 
                     'run', 'mri_info_visitid', 'dmri_dti_postqc_qc',
       'iqc_t2_ok_ser', 'iqc_mid_ok_ser', 'iqc_sst_ok_ser',
       'iqc_nback_ok_ser', 'tfmri_mid_beh_perform.flag',
       'tfmri_nback_beh_perform.flag', 'tfmri_sst_beh_perform.flag',
       'tfmri_mid_all_beta_dof', 'tfmri_mid_all_sem_dof',
       'tfmri_sst_all_beta_dof', 'tfmri_sst_all_sem_dof',
       'tfmri_nback_all_beta_dof', 'tfmri_nback_all_sem_dof',
       'mrif_score', 'mrif_hydrocephalus', 'mrif_herniation',
       'mr_findings_ok', 'tbl_numtrs', 'tbl_dof', 'tbl_nvols', 'tbl_tr', 'tbl_subthresh.nvols',
                     'rsfmri_cor_network.gordon_tr', 'rsfmri_cor_network.gordon_numtrs',
       'rsfmri_cor_network.gordon_nvols',
       'rsfmri_cor_network.gordon_subthresh.nvols',
       'rsfmri_cor_network.gordon_subthresh.contig.nvols',
       'rsfmri_cor_network.gordon_ntpoints', 'dataset_id_y', 'tbl_mean.motion', 'tbl_mean.trans', 'tbl_mean.rot',
       'tbl_max.motion', 'tbl_max.trans', 'tbl_max.rot']
    meta_cols = raw_df.columns[raw_df.columns.isin(base_meta_cols)].values
    metric_cols = raw_df.columns[~raw_df.columns.isin(base_meta_cols)].values
    
    mapper = DataFrameMapper([([nv],preprocessing.StandardScaler()) for nv in metric_cols])
    
    if (dn == 0) or (n_draws is not None):
        # Norm columns for variance estimation
        variance_mapper = DataFrameMapper([([nv],preprocessing.StandardScaler()) for nv in (list(metric_cols) + ['interview_age'])])
        var_df = raw_df.copy(deep=True)
        var_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(raw_df)

        print("Estimate variance contributions for each metric", flush=True)
        var_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_variance_metric_perm)(pn, perms[pn], metric, var_df) for metric in metric_cols)
    else:
        var_res = None

    #var_res = run_variance_perm(pn, perms[pn], raw_df, metric_cols)
    
    # set up perm
    raw_df.loc[:, 'unique_scanner'] = raw_df.loc[perms[pn], 'unique_scanner'].values

    # draw samples
    strata = ['unique_scanner']
    balance = ['gender', 'ehi_ss_score']
    order = ['interview_age']
    keys = ['subjectkey', 'interview_date']
    n_splits = 3
    ab_n_splits = 2
    draws_df = bal_samp(raw_df, strata, balance, order, keys, n_splits=n_splits, n_draws=n_draws)

    strata = ['unique_scanner']
    balance = ['gender', 'ehi_ss_score','age_3mos']
    order = ['interview_age']
    keys = ['subjectkey', 'interview_date']
    df_ab = raw_df.copy(deep=True)
    df_ab['age_3mos'] = (df_ab['interview_age'] // 3) * 3

    count_by_level = df_ab.groupby(strata+balance)[['subjectkey']].nunique().reset_index()
    levels_counts = count_by_level[count_by_level.subjectkey >= ab_n_splits].groupby(balance)[strata].nunique().reset_index().rename(columns={'unique_scanner':'sufficient_sites'})
    levels_counts.loc[levels_counts.sufficient_sites == levels_counts.sufficient_sites.max()]
    df_ab = df_ab.reset_index().merge(levels_counts, how='left', on=['gender','ehi_ss_score', 'age_3mos']).set_index('index')
    df_ab = df_ab.loc[df_ab.sufficient_sites == df_ab.sufficient_sites.max(),:]
    ab_draws_df = bal_samp(df_ab, strata, balance, order, keys, n_splits=ab_n_splits, n_draws=n_draws)

    # Create cv folds for gender age balance
    cv_splits = get_splits(draws_df, n_draws=n_draws, dn=dn)
    
    # Build run list
    run_params = [
    (raw_df.loc[cv[0], :],
              ymapper.transform(raw_df.loc[cv[0], :]).ravel(),
                raw_df.loc[cv[1], :],
                ymapper.transform(raw_df.loc[cv[1], :]).ravel(),
                pn, fn, name, metric_cols) for fn,cv in enumerate(cv_splits) for name in ['normal', 'age_rsd', 'cbagersd', 'combat']]
    if do_ab:
        # Prep data for age_balance
        ab_cv_splits = get_splits(ab_draws_df, n_draws=n_draws, dn=dn)
        run_params.extend([(raw_df.loc[cv[0], :],
                        ymapper.transform(raw_df.loc[cv[0], :]).ravel(),
                        raw_df.loc[cv[1], :],
                        ymapper.transform(raw_df.loc[cv[1], :]).ravel(),
                        pn, fn, name, metric_cols) for fn,cv in enumerate(ab_cv_splits) for name in ['ab_normal', 'ab_combat']])
    
    print("Fitting %d models"%len(run_params), flush=True)
    res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(per_fold)(*params) for params in run_params)
    
    if (dn == 0) or (n_draws is not None):
        # Combat for Variance
        combat_res = run_combat(raw_df.loc[:,metric_cols], raw_df.loc[:, ['interview_age', 'gender', 'ehi_ss_score', 'unique_scanner']])
        cb_df = raw_df.copy(deep=True)
        cb_df.loc[:, metric_cols] = combat_res[0].T
        try:
            cb_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(cb_df)
        except ValueError:
            cb_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(cb_df.fillna(cb_df.mean()))
        print("Estimate variance contributions for each metric after combat correction", flush=True)
        cb_var_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_variance_metric_perm)(pn, perms[pn], metric, cb_df) for metric in metric_cols)
    else:
        cb_var_res = None
    all_res = (res, var_res, cb_var_res)
    with open(out_path, 'wb') as h:
        pickle.dump(all_res, h)