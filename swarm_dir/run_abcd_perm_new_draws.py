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
from scipy.sparse.linalg import cg
from scipy import stats
import patsy

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
    

def run_combat(feats, meta, model="~interview_age + gender", par_prior=False, par_fallback=True):

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    robjects.r('library(sva)')
    robjects.r('library(BiocParallel)')
    combat = robjects.r('ComBat')
    model_matrix = patsy.dmatrix(model, meta)
    fmat = np.array(feats).T
    rbatch = robjects.IntVector(pd.Categorical(meta.unique_scanner).codes)
    combat_result = combat(dat = fmat, batch = rbatch, mod = model_matrix, par_prior=par_prior)
    combat_result = np.array(combat_result)
    if par_fallback & ~par_prior & (pd.isnull(combat_result).sum() > 0):
        print("Nonparametric prior failed, falling back to parametric.", flush=True)
        combat_result = combat(dat = fmat, batch = rbatch, mod = model_matrix, par_prior=True)
        combat_result = np.array(combat_result)
        
    if pd.isnull(combat_result).sum() > 0:
        raise ValueError("ComBat has returned nans")
        
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


#ModPermRes = namedtuple("ModPermRes",['pn','fn','name', 'clf', 'ho_score','cfn'])
#VarPermRes = namedtuple("VarPermRes", ["pn", "metric", "int_r2", "agh_r2", "aghs_r2", "aghss_r2", "aghsss_r2"])


def fit_model(X_r, Y_r, X_e, Y_e, pn, fn, name, mapper):
    clf = Pipeline([('preprocessing', mapper),
                    ('clf', LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 10000,
                                               penalty='l2',
                                               C=1,
                                               fit_intercept=True))])

    clf.fit(X_r, Y_r)

    res = {'pn':pn,
               'fn':      fn,
               'name':      name,
               'clf':      clf.named_steps['clf'],
               'ho_score':      clf.score(X_e, Y_e),
               'cfn':      confusion_matrix(Y_e, clf.predict(X_e))}
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
        cb_meta_cols = ['unique_scanner', 'interview_age', 'gender']
        X_r_cb = X_r_rsd
        X_e_cb = X_e_rsd
        # Fit combat seperately on training and test splits
        combat_res_r = run_combat(X_r_cb.loc[:, metric_cols], X_r_cb.loc[:, cb_meta_cols])
        X_r_cb.loc[:, metric_cols] = combat_res_r.T
        combat_res_e = run_combat(X_e_cb.loc[:, metric_cols], X_e_cb.loc[:, cb_meta_cols])
        X_e_cb.loc[:, metric_cols] = combat_res_e.T
        return fit_model(X_r_cb, Y_r, X_e_cb, Y_e, pn, fn, name, mapper)
    
    # Learn combat on training, apply to test, fit on x_r_cb, score on x_e_cb
    elif 'combat' in name:
        print("combat", name)
        cb_meta_cols = ['unique_scanner', 'interview_age', 'gender']
        X_r_cb = X_r.copy(deep=True)
        X_e_cb = X_e.copy(deep=True)
        combat_res_r = run_combat(X_r_cb.loc[:, metric_cols], X_r_cb.loc[:, cb_meta_cols])
        X_r_cb.loc[:, metric_cols] = combat_res_r.T
        combat_res_e = run_combat(X_e_cb.loc[:, metric_cols], X_e_cb.loc[:, cb_meta_cols])
        X_e_cb.loc[:, metric_cols] = combat_res_e.T
        return (fit_model(X_r_cb, Y_r, X_e_cb, Y_e, pn, fn, name, mapper))

    
def cg_reg(mdl):
    """Fit a statsmodel model with conjugate gradient iteration.
       Parametrs
       mdl: a statsmodel model object
       Returns:
       betas, rsquared, rescode
       beta: the converged solution
       rsquared: pearson rsquared of the solution
       rescode: 0 : successful exit >0 : convergence to tolerance not achieved, number of iterations <0 : illegal input or breakdown"""
    X = mdl.exog
    Y = mdl.endog

    A = np.matmul(X.T, X)
    b = np.matmul(X.T, Y)

    betas, rescode = cg(A, b, tol=1e-9)
    
    rsquared = stats.pearsonr(np.matmul(X,betas), Y)[0]**2
    return betas, rsquared, rescode

    
def run_variance_metric_perm(pn, perm, metric, df, include_models=False, residualize=False, cg=True):
    # Don't forget to z-score df before this step
    df = df.copy()
    if perm is not None:
        df['perm_metric'] = df.copy().loc[perm, metric].values
    else:
        df['perm_metric'] = df.copy().loc[:, metric].values
    res = {}
    if residualize:
        int_fit = smf.ols('perm_metric ~ 1 ', data = df).fit()
        agh_fit = smf.ols('perm_metric ~ 1 + interview_age + gender', data = df).fit()
        df['perm_metric'] = agh_fit.resid
        res['agh_resid_var'] = df.perm_metric.var()
        aghs_fit = smf.ols('perm_metric ~ 1 + mri_info_manufacturer', data = df).fit()
        df['perm_metric'] = aghs_fit.resid
        res['aghs_resid_var'] = df.perm_metric.var()
        aghss_fit = smf.ols('perm_metric ~ 1 + mri_info_manufacturersmn', data = df).fit()
        df['perm_metric'] = aghss_fit.resid
        res['aghss_resid_var'] = df.perm_metric.var()
        aghsss_fit = smf.ols('perm_metric ~ 1 + unique_scanner', data = df).fit()
    else:
        int_mod = smf.ols('perm_metric ~ 1', data=df)
        agh_mod = smf.ols('perm_metric ~ interview_age + gender', data = df)
        aghs_mod = smf.ols('perm_metric ~ interview_age + gender + mri_info_manufacturer', data = df)
        aghss_mod =smf.ols('perm_metric ~ interview_age + gender + mri_info_manufacturer + mri_info_manufacturersmn', data = df)
        aghsss_mod =smf.ols('perm_metric ~ interview_age + gender + mri_info_manufacturer + mri_info_manufacturersmn + unique_scanner', data = df)
        if cg:
            _, int_rsquared, res['int_rescode'] = cg_reg(int_mod)
            # we've mean centered 
            if pd.isnull(int_rsquared):
                int_rsquared = 0
            _, agh_rsquared, res['agh_rescode'] = cg_reg(agh_mod)
            _, aghs_rsquared, res['aghs_rescode'] = cg_reg(aghs_mod)
            _, aghss_rsquared, res['aghss_rescode'] = cg_reg(aghss_mod)
            _, aghsss_rsquared, res['aghsss_rescode'] = cg_reg(aghsss_mod)
        else:
            int_fit = int_mod.fit()
            int_rsquared = int_fit.rsquared
            agh_fit = agh_mod.fit()
            agh_rsquared = agh_fit.rsquared
            aghs_fit = aghs_mod.fit()
            aghs_rsquared = aghs_fit.rsquared
            aghss_fit = aghss_mod.fit()
            aghss_rsquared = aghss_fit.rsquared
            aghsss_fit = aghsss_mod.fit()
            aghsss_rsquared = aghsss_fit.rsquared
            
    res.update({'pn':pn,
           'metric': metric,
           'variance': df.perm_metric.var(),
           'int_r2': int_rsquared,
           'agh_r2': agh_rsquared,
           'aghs_r2': aghs_rsquared,
           'aghss_r2': aghss_rsquared,
           'aghsss_r2': aghsss_rsquared})
    if include_models:
        res.update({'int_mod' : int_mod,
                    'agh_mod' : agh_mod,
                    'aghs_mod': aghs_mod,
                    'aghss_mod': aghss_mod,
                    'aghsss_mod': aghsss_mod})
    return res


def run_variance_perm(pn, perm, df, metrics):
    return [run_variance_metric_perm(pn, perm, metric, df) for metric in metrics]

def get_mod_res(fit):
    try:
        sum2t = fit.summary2().tables
    except AttributeError:
        sum2t = fit.summary().tables
    
    stretched = []
    for var, row in sum2t[1].iterrows():
        for name,val in row.iteritems():
            stretched.append({'param':var, 'var':name, 'val':val})
    stretched = pd.DataFrame(stretched).reset_index(drop=True)
    
    try:
        mod_res = (pd.concat([sum2t[0].loc[:,[0,1]],
                              sum2t[0].loc[:,[2,3]].rename(columns={2:0,3:1}),
                              sum2t[2].loc[:,[0,1]],
                              sum2t[2].loc[:,[2,3]].rename(columns={2:0,3:1})])
                   .reset_index(drop=True)
                   .rename(columns={0:'var', 1:'val'})).reset_index(drop=True)
    except IndexError:
        mod_res = (pd.concat([sum2t[0].loc[:,[0,1]],
                              sum2t[0].loc[:,[2,3]].rename(columns={2:0,3:1})])
                   .reset_index(drop=True)
                   .rename(columns={0:'var', 1:'val'})).reset_index(drop=True)
    mod_res = pd.concat([mod_res, stretched], sort=False).reset_index(drop=True)
    return mod_res

def fit_agh_mod(df):
    agh_mod = smf.ols('perm_metric ~ 1 + interview_age + gender', data = df)
    agh_fit = agh_mod.fit()
    return get_mod_res(agh_fit)

def calc_sig(pn, perm, metric, df):
    if perm is not None:
        df['perm_metric'] = df.copy().loc[perm, metric].values
    else:
        df['perm_metric'] = df.copy().loc[:, metric].values
    res = {}
    me_mdl = smf.mixedlm('perm_metric ~ 1 + interview_age + gender', re_formula = '~1', groups='unique_scanner', data=df)
    me_fit = me_mdl.fit()
    res['pn'] = pn
    res['metric'] = metric
    res['me_fit'] = get_mod_res(me_fit)
    all_ttest = stats.ttest_1samp(df.perm_metric, 0)
    ttest_res = [{'pn':pn,
                  'metric': metric,
                  'var': 't',
                  'val': all_ttest.statistic,
                  'unique_scanner': "all",
                  'method':'ttest',
                  'param': 'Intercept'},
                {'pn':pn,
                  'metric': metric,
                  'var': 'P>|t|',
                  'val': all_ttest.pvalue,
                  'unique_scanner': "all",
                  'method':'ttest',
                  'param': 'Intercept'}]
    site_ttest = df.groupby('unique_scanner').perm_metric.apply(lambda x: stats.ttest_1samp(x, 0))

    for ii, tt in site_ttest.iteritems():
        ttest_res.extend([{'pn':pn,
                      'metric': metric,
                      'var': 't',
                      'val': tt.statistic,
                      'unique_scanner': ii,
                      'method':'ttest',
                      'param': 'Intercept'},
                    {'pn':pn,
                      'metric': metric,
                      'var': 'P>|t|',
                      'val': tt.pvalue,
                      'unique_scanner': ii,
                      'method':'ttest',
                      'param': 'Intercept'}])
    ttest_res = pd.DataFrame(ttest_res)
    res['ttest_res'] = ttest_res
    res['ols_all_fit'] = fit_agh_mod(df)
    res['ols_site_fit'] = df.groupby('unique_scanner').apply(fit_agh_mod)
    return res
    
def consolidate_sig_res(res):
    ols_fits = []
    for rr in res:
        rr['me_fit']['pn'] = rr['pn']
        rr['me_fit']['metric'] = rr['metric']
        rr['me_fit']['unique_scanner'] = "all"
        rr['me_fit']['method'] = 'me'

        rr['ols_all_fit']['unique_scanner'] = "all"
        rr['ols_all_fit']['method'] = 'ols'
        rr['ols_site_fit'] = rr['ols_site_fit'].reset_index().drop(columns='level_1')
        rr['ols_site_fit']['method'] = 'ols'

        ols_df = pd.concat([rr['ols_all_fit'], rr['ols_site_fit'], rr['me_fit'], rr['ttest_res']], sort=False, ignore_index=True)
        ols_df['pn'] = rr['pn']
        ols_df['metric'] = rr['metric']
        ols_fits.append(ols_df)
    ols_fits = pd.concat(ols_fits, sort=False, ignore_index=True)
    return ols_fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('pn', type=int, help="Integer specifying which permutation or bootstrap to run")
    parser.add_argument('perm_path', type=str, help="Path to pickle specifying permuted or bootstrap indices")
    parser.add_argument('ymapper_path', type=str, help="Path to pickled ymapper")
    parser.add_argument('raw_df_path', type=str, help="Path to pickled raw dataframe")
    #parser.add_argument('draws_df_path', type=str, help="Path to pickled draw dataframe")

    parser.add_argument('out_path', type=str, help="Path dump pickle")
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--n_draws', type=int, action='store')
    parser.add_argument('--dn', type=int, action='store')
    parser.add_argument('--do_ab', action='store_true')
    parser.add_argument('--bootstrap', action='store_true', help="Is this a bootstrap or a permutation")
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
    bootstrap = args.bootstrap

    # Pick out meta columns and metric columns
    base_meta_cols = ['contrast', 'fmri_beta_gparc_numtrs', 'fmri_beta_gparc_tr', 'lmt_run',
                      'mid_beta_seg_dof', 'task', 'collection_id', 'dataset_id', 'subjectkey',
                      'src_subject_id', 'interview_date', 'interview_age',
                      'gender', 'event_name', 'visit', 'rsfm_tr', 'eventname',
                      'rsfm_nreps', 'rsfm_numtrs', 'pipeline_version',  'scanner_manufacturer_pd',
                      'scanner_type_pd', 'mri_info_deviceserialnumber', 'magnetic_field_strength',
                      'procdate', 'collection_title', 'promoted_subjectkey', 'study_cohort_name',
                      'ehi_ss_score', '_merge', 'qc_ok', 'age_3mos', 'age_6mos', 
                      'abcd_betnet02_id', 'fsqc_qc',
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
                      'rsfmri_cor_network.gordon_ntpoints', 'dataset_id_y',
                      'tbl_mean.motion', 'tbl_mean.trans', 'tbl_mean.rot',
                      'tbl_max.motion', 'tbl_max.trans', 'tbl_max.rot',
                      'rsfmri_cor_network.gordon_mean.motion',
                      'rsfmri_cor_network.gordon_max.motion',
                      'rsfmri_cor_network.gordon_mean.trans',
                      'rsfmri_cor_network.gordon_max.trans',
                      'rsfmri_cor_network.gordon_mean.rot',
                      'rsfmri_cor_network.gordon_max.rot',
                      'index', 'cr']
    meta_cols = raw_df.columns[raw_df.columns.isin(base_meta_cols)].values
    metric_cols = raw_df.columns[~raw_df.columns.isin(base_meta_cols)].values
    
    mapper = DataFrameMapper([([nv],preprocessing.StandardScaler()) for nv in metric_cols])
    
    if bootstrap:
        bs_df = raw_df.loc[perms[pn], :].reset_index(drop=True)
        nobs_df = raw_df.loc[(~raw_df.subjectkey.isin(bs_df.subjectkey.unique())), :].reset_index(drop=True)
        
    if (dn == 0) or (n_draws is not None):
        # Norm columns for variance estimation
        variance_mapper = DataFrameMapper([([nv],preprocessing.StandardScaler()) for nv in (list(metric_cols) + ['interview_age'])])
        if bootstrap:
              var_df = bs_df.copy(deep=True)
        else:
            var_df = raw_df.copy(deep=True)
#         var_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(var_df)

        print("Estimate variance contributions for each metric", flush=True)
        if bootstrap:
            var_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_variance_metric_perm)(pn, None, metric, var_df) for metric in metric_cols)
            sig_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_sig)(pn, None, metric, var_df) for metric in metric_cols)
        else:
            var_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_variance_metric_perm)(pn, perms[pn], metric, var_df) for metric in metric_cols)
            sig_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_sig)(pn, perms[pn], metric, var_df) for metric in metric_cols)
        sig_res = consolidate_sig_res(sig_res)
        sig_res['combat'] = False
    else:
        var_res = None
        sig_res = None
    if bootstrap:
        # draw samples
        strata = ['unique_scanner']
        balance = ['gender', 'ehi_ss_score']
        order = ['interview_age']
        keys = ['subjectkey', 'interview_date']
        n_splits = 1
        ab_n_splits = 1
        draws_df = bal_samp(bs_df, strata, balance, order, keys, n_splits=n_splits, n_draws=n_draws)
        draw_ind = draws_df.draw_0[draws_df.draw_0.notnull()].index.values
        draws_df_nobs = bal_samp(nobs_df, strata, balance, order, keys, n_splits=n_splits, n_draws=n_draws)
        draw_ind_nobs = draws_df_nobs.draw_0[draws_df_nobs.draw_0.notnull()].index.values
        
        run_params = [
                      (bs_df.loc[draw_ind, :],
                      ymapper.transform(bs_df.loc[draw_ind, :]).ravel(),
                      nobs_df.loc[draw_ind_nobs, :],
                      ymapper.transform(nobs_df.loc[draw_ind_nobs, :]).ravel(),
                      pn, 0, name, metric_cols) for name in ['normal', 'age_rsd', 'cbagersd', 'combat']]
        
        if do_ab:
            strata = ['unique_scanner']
            balance = ['gender', 'ehi_ss_score','age_6mos']
            order = ['interview_age']
            keys = ['subjectkey', 'interview_date']
            df_ab = bs_df.copy(deep=True)
            df_ab['age_6mos'] = (df_ab['interview_age'] // 6) * 6
            
            count_by_level = df_ab.groupby(strata+balance)[['subjectkey']].nunique().reset_index()
            levels_counts = count_by_level[count_by_level.subjectkey >= ab_n_splits].groupby(balance)[strata].nunique().reset_index().rename(columns={'unique_scanner':'sufficient_sites'})
            levels_counts.loc[levels_counts.sufficient_sites == levels_counts.sufficient_sites.max()]
            df_ab = df_ab.reset_index().merge(levels_counts, how='left', on=['gender','ehi_ss_score', 'age_6mos']).set_index('index')
            df_ab = df_ab.loc[df_ab.sufficient_sites == df_ab.sufficient_sites.max(),:]
            ab_draws_df = bal_samp(df_ab, strata, balance, order, keys, n_splits=ab_n_splits, n_draws=n_draws)
            ab_draw_ind = ab_draws_df.draw_0[ab_draws_df.draw_0.notnull()].index.values
            
            df_ab_nobs = nobs_df.copy(deep=True)
            df_ab_nobs['age_6mos'] = (df_ab_nobs['interview_age'] // 6) * 6
            
            count_by_level = df_ab_nobs.groupby(strata+balance)[['subjectkey']].nunique().reset_index()
            levels_counts = count_by_level[count_by_level.subjectkey >= ab_n_splits].groupby(balance)[strata].nunique().reset_index().rename(columns={'unique_scanner':'sufficient_sites'})
            levels_counts.loc[levels_counts.sufficient_sites == levels_counts.sufficient_sites.max()]
            df_ab_nobs = df_ab_nobs.reset_index().merge(levels_counts, how='left', on=['gender','ehi_ss_score', 'age_6mos']).set_index('index')
            df_ab_nobs = df_ab_nobs.loc[df_ab_nobs.sufficient_sites == df_ab_nobs.sufficient_sites.max(),:]
            ab_draws_df_nobs = bal_samp(df_ab_nobs, strata, balance, order, keys, n_splits=ab_n_splits, n_draws=n_draws)
            ab_draw_ind_nobs = ab_draws_df_nobs.draw_0[ab_draws_df_nobs.draw_0.notnull()].index.values

            
            run_params.extend([
                      (df_ab.loc[ab_draw_ind, :],
                      ymapper.transform(df_ab.loc[ab_draw_ind, :]).ravel(),
                      df_ab_nobs.loc[ab_draw_ind_nobs, :],
                      ymapper.transform(df_ab_nobs.loc[ab_draw_ind_nobs, :]).ravel(),
                      pn, 0, name, metric_cols) for name in  ['ab_normal', 'ab_combat']])
    else:
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
            strata = ['unique_scanner']
            balance = ['gender', 'ehi_ss_score','age_6mos']
            order = ['interview_age']
            keys = ['subjectkey', 'interview_date']
            df_ab = raw_df.copy(deep=True)
            df_ab['age_6mos'] = (df_ab['interview_age'] // 6) * 6

            count_by_level = df_ab.groupby(strata+balance)[['subjectkey']].nunique().reset_index()
            levels_counts = count_by_level[count_by_level.subjectkey >= ab_n_splits].groupby(balance)[strata].nunique().reset_index().rename(columns={'unique_scanner':'sufficient_sites'})
            levels_counts.loc[levels_counts.sufficient_sites == levels_counts.sufficient_sites.max()]
            df_ab = df_ab.reset_index().merge(levels_counts, how='left', on=['gender','ehi_ss_score', 'age_6mos']).set_index('index')
            df_ab = df_ab.loc[df_ab.sufficient_sites == df_ab.sufficient_sites.max(),:]
            ab_draws_df = bal_samp(df_ab, strata, balance, order, keys, n_splits=ab_n_splits, n_draws=n_draws)

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
        if bootstrap:
            combat_res = run_combat(bs_df.loc[:,metric_cols], bs_df.loc[:, ['interview_age', 'gender', 'unique_scanner']])
            cb_df = bs_df.copy(deep=True)
            cb_df.loc[:, metric_cols] = combat_res.T     
        else:
            cb_df = raw_df.copy(deep=True)
            # for combat, since raw_df is already permed, also perm other predictors, leave metrics alone
            predictors = ['interview_age', 
                          'gender', 
                          'mri_info_manufacturer', 
                          'mri_info_manufacturersmn']
            cb_df.loc[:, predictors] = cb_df.loc[perms[pn], predictors].values
            combat_res = run_combat(cb_df.loc[:,metric_cols], cb_df.loc[:, ['interview_age', 'gender', 'unique_scanner']])
            cb_df.loc[:, metric_cols] = combat_res.T
            
#         try:
#             cb_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(cb_df)
#         except ValueError:
#             cb_df.loc[:, list(metric_cols) + ['interview_age']] = variance_mapper.fit_transform(cb_df.fillna(cb_df.mean()))
        print("Estimate variance contributions for each metric after combat correction", flush=True)

        cb_var_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_variance_metric_perm)(pn, None, metric, cb_df) for metric in metric_cols)
        cb_sig_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_sig)(pn, None, metric, cb_df) for metric in metric_cols)
        cb_sig_res = consolidate_sig_res(cb_sig_res)
        cb_sig_res['combat'] = True
        sig_res = pd.concat([sig_res, cb_sig_res], sort=False, ignore_index=True)
    else:
        cb_var_res = None
        cb_sig_res = None
    all_res = (res, var_res, cb_var_res)
    with open(out_path, 'wb') as h:
        pickle.dump(all_res, h)
    sig_res.to_pickle(out_path.replace('.pkz', '_sig.gz'))