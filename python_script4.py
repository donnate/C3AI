''' Module class for training the EM algorithm
'''
import copy
from datetime import datetime 
import json
import numpy as np
import pandas as pd
import pickle
import scipy as sc
import sys

import requests
import json
from datetime import datetime, timedelta
from scipy.stats import beta 

lookup = pd.read_csv("full_db.csv")
dict_v = pickle.load(open("params.pkl", 'rb'))
SPECIFICITY_PRIORS = dict_v['SPECIFITCITY_PRIORS']
SENSITIVITY_PRIORS = dict_v['SENSITIVITY_PRIORS']
CONTEXT_INFO = dict_v['CONTEXT_INFO' ]
SYMPTOMS = ['cough',
 'fever',
 'sob',
 'diarrhea',
 'fatigue',
 'headache',
 'loss_of_smell',
 'loss_of_taste',
 'runny_nose',
 'myalgia',
 'sore_throat']
SYMPTOMS_QUANT = dict_v['SYMPTOMS_QUANT']
hyperparams = dict_v['HYPERPARAMS']
group_tests = dict_v['group_tests']


def moment_matching_beta(xbar, s2):
    """ Computes the parameters (for moment matching)
    for a beta distribution from a given mean and variance
    INPUTS:
    ------------------------------------------------------------
    xbar                    : mean
    s2                      : desired variance
    """
    a = xbar * (xbar * (1. - xbar) / s2 - 1.0)
    b = a * (1. - xbar) / xbar
    return([a,b])
    
  
def MAP_beta(X1,X0, alpha_0, beta_0):
    """ the Maximum A Posteriori estimate for the beta
    distribution
    INPUTS:
    ------------------------------------------------------------
    X1                       : scores for the "1"s observed values
    X0                       : scores for the "0"s observed values
    alpha_0, beta_0          : hyper parameters for the beta distribution
    """
    mu_map = (np.sum(X1) + alpha_0 - 1) / (np.sum(X0) + beta_0 - 2)
    return mu_map



class EMClassifier3:
    ''' Parent class for EM variations
    Variations will have to instantiate the quantitative symptom functions
    and potentially overwrite the ones for the symptoms if doing SEM
    Parameters
    ----------
    method_missing_T : str, {'truncated', 'latent-DT', 'latent-DT-newprior'}
        Method chosen to imput the missing T:
        - 'truncated': cf Section "StEM with missing T: truncated Bayesian network"
            * E-step: Imput Di
                - When T non-missing: using the full Bayesian network
                - When T missing: using the truncated Bayesian network without the T branch
            * M-step:
                - Update p of symptomatic/symptoms/betas as before, with imputed Dis
                - Update sensitivity/specificity by summing only on i's with non-missing T
        - 'latent-DT': cf Section "StEM with missing T: missing and hidden variables"
            * E-step: Imput Di, Ti
                - When T non-missing: as usual with the full Bayesian network
                - When T missing: using the joint posterior p(D, T| observations)
            * M-step: Update parameters using imputed Di and Ti
        - 'latent-DT-newprior': cf Section "StEM with missing T: missing and hidden variables" (Enhancement)
            * Same as 'latent-DT' but using a different prior on sensitivity/specificity 
            when T is missing.
    '''
    def __init__(self, list_symptoms=SYMPTOMS,
                 list_symptoms_quant=SYMPTOMS_QUANT,
                 list_context_variables=CONTEXT_INFO,
                 params_dist=None,
                 group_tests = group_tests,
                 hyperparams_dist=None, EM_steps=20, B=1000, beta=None,
                 specificity_priors=SPECIFICITY_PRIORS,
                 sensitivity_priors=SENSITIVITY_PRIORS,
                 separate_asymptomatic=False,
                 update_sensitivity=False,
                 update_symptomatic=False,
                 update_prior_symptoms=True,
                 completely_agnostic_prior=False,
                 convergence_threshold=1e-3,
                 params_names_quant=['a', 'b'],
                 use_prevalence = True,
                 use_prevalence_ili = False,
                 stochastic = False,
                 method_missing_T='latent-DT'):
        self.model = None
        self.EM_steps = EM_steps  ## nb of EM steps to take
        self.separate_asymptomatic = separate_asymptomatic ### are we taking into account a symptomatic variable? Or should this be fixed?
        self.update_sensitivity = update_sensitivity ### are we updating the sensitivity of the tests?
        self.update_symptomatic = update_symptomatic
        self.use_prevalence = use_prevalence
        self.use_prevalence_ili = use_prevalence_ili
        self.test_groups = group_tests
        self.B = B             
        self.list_context_variables = list_context_variables
        self.list_symptoms = list_symptoms
        self.list_symptoms_quant = list_symptoms_quant.keys()
        self.specificity = specificity_priors
        self.sensitivity = sensitivity_priors
        self.stochastic = stochastic
        self.method_missing_T = method_missing_T
        self.convergence = 0
        self.convergence_threshold = convergence_threshold
        self.completely_agnostic_prior=completely_agnostic_prior
        self.update_prior_symptoms = update_prior_symptoms
        self.params_names_quant =  params_names_quant  ### names of the parameters for the quant distributions --allows  extra flexibility
        self.H = np.diagflat(np.zeros((len(list_context_variables)+1)))
        self.variability_prev = 1.0
        
        if beta is None:
            self.beta_reg = np.zeros(len(self.list_context_variables)) ##coef for the context log reg
        else:
            self.beta_reg = np.zeros(len(self.list_context_variables))
        ##### Initialize all the parameters in the Bayesian Model
        list_params_names = list(self.specificity.keys()) + list(self.sensitivity.keys()) ### tests
        list_params_names += [a  +  n + '_' + e for a in ['alpha_', 'beta_', 'p_']
                            for e in ['0','1'] for n in self.list_symptoms + ['symptomatic']] ### symptoms
        self.hyperparams = {k: 1.2 for k in list_params_names}
        self.hyperparams.update({'mu_reg': np.zeros(1 + len(CONTEXT_INFO)),
                                 'a_reg': 10,
                                 'Lambda_reg': np.diag(np.ones(1 + len(CONTEXT_INFO))),
                                 'b_reg': 10})  ###  regression
        self.hyperparams.update({ a + '_' + n + '_' + e: np.ones(list_symptoms_quant[n][0])
                            for n in self.list_symptoms_quant 
                            for a in list_symptoms_quant[n][1] 
                            for e in ['0','1']})
        self.params = {k: 0.5 for k in list_params_names}  ## simple declaration
        self.params.update({'mu_reg': np.zeros(1 + len(CONTEXT_INFO)),
                            'a_reg': 10,
                            'Lambda_reg': np.diag(np.ones(len(CONTEXT_INFO))),
                            'b_reg': 10}) ###  regression
        self.params.update({a + '_' + k + '_' + e: 0.5* np.ones(list_symptoms_quant[k][0])
                            for k in list_symptoms_quant.keys()
                            for e in ['0', '1'] for a in list_symptoms_quant[k][1]})
        self.hyperparams['alpha_T_0'] = self.specificity['alpha_T_0'] 
        self.hyperparams['beta_T_0'] = self.specificity['beta_T_0']
        self.params['alpha_T_0'] = self.specificity['alpha_T_0'] 
        self.params['beta_T_0'] = self.specificity['beta_T_0']
        self.params['p_T_0'] = self.specificity['alpha_T_0']/(self.specificity['alpha_T_0'] +self.specificity['beta_T_0'])
        for k in self.sensitivity.keys():
            self.hyperparams[k] = self.sensitivity[k] 
            self.params[k] = self.sensitivity[k]


        ##### Option to pass different more informative initial values1
        if hyperparams_dist is not None:
          print(np.intersect1d(list(self.hyperparams.keys()),list(hyperparams_dist.keys())))
          for k in np.intersect1d(list(self.hyperparams.keys()),list(hyperparams_dist.keys())):
            self.hyperparams[k] = hyperparams_dist[k]
        if params_dist is not None:
          for k in np.intersect1d(list(self.params.keys()),list(params_dist.keys())):
            self.params[k] = params_dist[k]
    
    def update_logs_bernouilli(self, X, n):
      '''' Updates the log likelihood for observations X (dim N),
      with the params of the model
      '''
      if 'T_' in n:
          res = (X) * np.log(self.params['p_T_0']) \
              + (1. - X) * np.log(1-self.params['p_T_0'])\
              - (X) * np.log(self.params['p_' + n +'_1'])\
              - (1. - X) * np.log(1-self.params['p_' + n +'_1'])
      else:
        res = (X) * np.log(self.params['p_' + n +'_0']) \
              + (1. - X) * np.log(1-self.params['p_' + n +'_0'])\
              - (X) * np.log(self.params['p_' + n +'_1'])\
              - (1. - X) * np.log(1-self.params['p_' + n +'_1'])
    
      if self.method_missing_T == 'truncated':
          # This takes into account the case 
          # where T is missing, i.e. T=NaN in the df.
          # in the truncated Bayesian model
          res[np.isnan(X)] = 0
      return res

    def update_logs_multinomial(self, X, n):
        res0 = np.array(list(map(lambda x: np.log(self.params['pi_'+ n + '_0' ][int(x)]), X)))
        res1 = np.array(list(map(lambda x: np.log(self.params['pi_'+ n + '_1' ][int(x)]), X)))
        return res0 - res1
    
    def update_logs_samplebernouilli(self, X, sample):
      ''' Updates the log likelihood for observations X (dim N),
      with samples proba sample (N x B)
      '''
      res = (np.einsum('i, ij -> ij', X, np.log(sample)) \
            + np.einsum('i, ij -> ij', 1-X, np.log(1 - sample)))
      return res
    
    def add_updates2beta_prior(self, imputed_labels, X):
      ''' Computes the updates to add to the beta hyperparams
      '''
      # X may be NaN, for example if X=T
      add_one_alpha = np.multiply(imputed_labels, X)
      if self.method_missing_T == 'truncated':
          add_one_alpha = np.nan_to_num(add_one_alpha, nan=0.)

      add_one_beta = np.multiply(imputed_labels, 1-X)
      if self.method_missing_T == 'truncated':
          add_one_beta = np.nan_to_num(add_one_beta, nan=0.)

      # This only sums over the non-NaN values
      add_alpha = np.sum(add_one_alpha)
      add_beta = np.sum(add_one_beta)
      return add_alpha, add_beta
    
    def update_logs_beta(self, X, n):
      ''' Updates log likelihood for X if X is beta
      '''
      B1 = sc.special.beta(self.params['alpha_' + n +'_1'],
                          self.params['beta_' + n +'_1'])
      B0 = sc.special.beta(self.params['alpha_' + n +'_0'],
                          self.params['beta_' + n +'_0'])
      res = (self.params['alpha_' + n +'_0']-1) * np.log(X)+\
            (self.params['beta_' + n +'_0']-1) * np.log(1-X)+\
          -(self.params['alpha_' + n +'_1']-1) * np.log(X)+\
          -(self.params['beta_' + n +'_1']-1) * np.log(1-X)+\
          np.ones((len(index_sympt))) * (np.log(B1) - np.log(B0))
      return res

    def update_dirichlet(self, X, imputed_labels, n):
        added1 = [np.sum(np.multiply((X == k), imputed_labels))
                 for k in np.arange(self.params['alpha_dir_'+ n].shape[0])]
        self.params['pi_'+ n] = (self.hyperparams['alpha_dir_'+ n] + added1)/ np.sum(self.hyperparams['alpha_dir_'+ n] + added1)
        self.params['alpha_dir_'+ n] = (self.hyperparams['alpha_dir_'+ n] + added1)
                
    
    def update_bernouilli(self, n):
      ''' Updates params for X if X is bernouilli
      '''
      if (self.params['alpha_'+ n]> 1 and self.params['beta_'+ n]>1):
        res = ((self.params['alpha_'+ n] -1)/
              (self.params['alpha_'+ n]+ self.params['beta_'+ n] - 2))
      else:
        res = ((self.params['alpha_'+ n] )/
              (self.params['alpha_'+ n]+ self.params['beta_'+ n]))
      self.params['p_' + n]  = res
    
    def update_beta(self, n, add_alpha, add_beta):
      ''' Updates params for X if X is beta
      '''
      self.params['alpha_'+ n] = add_alpha + self.hyperparams['alpha_'+ n] 
      self.params['beta_'+ n] = add_beta + self.hyperparams['beta_'+ n]

    def update_mvn_known_Sig(self, n, X, Y):
      ''' Updates params for n if n is a multivariate normal distribution
      (ie, the regression coefficients for instance, assuming known covariance)
      '''
      #print("mvn start, nb of Y nan",np.sum(np.isnan(Y)))
      x = np.hstack([np.expand_dims(np.ones(X.shape[0]),1),
                     X[self.list_context_variables].values.astype(float)])   #### add intercept
      self.params['Lambda_'+ n] = self.hyperparams['Lambda_'+ n]  + x.T.dot(x)
      #print(x.T.dot(x))
      #print(self.params['Sigma_'+ n])
      #Om = np.linalg.inv(self.params['Sigma_'+ n])
      #print("inverse", np.linalg.inv(self.params['Sigma_'+ n]))
      hat_beta = np.linalg.inv((x.T.x).dot(Y))
      self.params['mu_'+ n] = np.linalg.inv(self.params['Lambda_'+ n]).dot(self.hyperparams['Lambda_'+ n].dot(self.hyperparams['mu_'+ n])\
                                                                          + x.T.dot(x.dot(hat_beta))) 
      #print("mu", self.params['mu_'+ n], self.hyperparams['mu_'+ n] , x.T.dot(Y))
      self.beta_reg = self.params['mu_'+ n] 

    def update_mvn(self, n, Y, X):
      ''' Updates params for n if n is a multivariate normal distribution
      (ie, the regression coefficients for instance) when the covariance is unknown
      '''
      x = np.hstack([np.expand_dims(np.ones(X.shape[0]),1),
                     X[self.list_context_variables].values.astype(float)])   #### add intercept
      N = X.shape[0]
      self.params['Lambda_'+ n] = self.hyperparams['Lambda_'+ n] + x.T.dot(x)
      self.params['mu_'+ n] = np.linalg.inv(self.params['Lambda_'+ n]).dot(self.hyperparams['Lambda_'+ n].dot(self.hyperparams['mu_'+ n])\
                                                                          + x.T.dot(Y)) 
      self.params['a_'+ n] = self.hyperparams['a_'+ n]  + N * 0.5
      self.params['b_'+ n] = self.hyperparams['b_'+ n] + 0.5 * Y.T.dot(Y) \
                            +  0.5 * (self.hyperparams['mu_'+ n].T.dot(self.hyperparams['Lambda_'+ n].dot((self.hyperparams['mu_'+ n])))) \
                            -  0.5 * (self.params['mu_'+ n].T.dot(self.params['Lambda_'+ n].dot((self.params['mu_'+ n])))) 
      #self.params['b_'+ n] = self.hyperparams['b_'+ n] + \
      # (Y - x.dot(self.params['mu_'+ n])).T.dot(Y - x.dot(self.params['mu_'+ n])) + \
      #(self.params['mu_'+ n] - self.hyperparams['mu_'+ n]).T.dot(self.params['Sigma_'+ n].dot((self.params['mu_'+ n] - self.hyperparams['mu_'+ n]))) 


      #(self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
      #print("mu", self.params['mu_'+ n], self.hyperparams['mu_'+ n] , x.T.dot(Y))
      #self.beta_reg = self.params['mu_'+ n] 
      #mean_data = np.mean(x, axis=0)
      #sum_squares = np.sum([np.array(np.matrix(xx - mean_data).T * np.matrix(xx - mean_data)) for xx in x], axis=0)
      
      


    
    def update_beta_prob_momentmatch(self, temp_lab, X, n):
      ''' Updates params for X if X is beta by moment matching
      '''
      xbar = np.sum(np.multiply(temp_lab , X)) / np.sum(temp_lab)
      s2 = np.sum(np.multiply(temp_lab , (X -xbar)**2)) / np.sum(temp_lab)
      a = xbar * (xbar * (1. - xbar) / s2 - 1.0)
      self.params['alpha_'+ n] = a
      self.params['beta_'+ n] =  a * (1. - xbar) / xbar
    
    def update_logs_quant(self, X, n):
      self.update_logs_multinomial(X, n)
      return None
    
    def update_quant_prior(self, X, imputed_labels, n):
      self.update_dirichlet(X, imputed_labels, n)
        
    def update_logs_prior(self, X, n):
        return self.update_logs_multinomial(X, n)
        
    def sample_quant(self, n, size):
        sample = np.random.dirichlet(self.params['alpha_dir_'+ n], size * self.B)
        samples = np.reshape(sample, ((size, self.B, sample.shape[1])))
        return samples

    def update_logs_samplequant(self, X, samples):
      enc = np.zeros((X.size, samples.shape[2]))
      enc[np.arange(X.size),X.astype('int')] = 1
      vlog = np.vectorize(lambda x: np.log(x) if x>1e-7 else np.log(1e-7))
      res = np.einsum('ik, ijk -> ij', enc, vlog(samples)) 
      return res
   
    def expectation(self, data, T):
        """ Computes the expectation of the hidden variables given
        the different parameters. The hidden variables in our model are D and self.x, self.y
        INPUTS
        ---------------------------------------------------------------------
        Y                   :       context + questionnaire data
        T                   :       image classification label (binary)
        D                   :       imputed diagnostic
        """
        #### MLE
        N = data.shape[0]
        index_sympt = np.arange(data.shape[0])
        log_odds = np.zeros(N)
        if self.use_prevalence:
            log_odds_prev = np.log(data['prevalence'].values)- np.log(1.0 - data['prevalence'].values )
            log_odds += - log_odds_prev
        elif self.use_prevalence_ili:
              log_odds_prev = np.multiply(data['symptomatic'].values, 
                                        np.log(data['p01'].values)- np.log(1.0 - data['p11'].values )
                            ) + np.multiply(1.0 - data['symptomatic'].values, 
                                        np.log(data['p00'].values)- np.log(1.0 - data['p10'].values )
                            )
              log_odds += log_odds_prev
        else:
            ### Now add the prior with context info
            self.beta_reg = self.params['mu_reg']
            Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                          data[self.list_context_variables].values.astype('float')])
            #Y = data[self.list_context_variables].values.astype('float')
            log_odds = - np.log(data['prevalence'].values) +  np.log(1.0 - data['prevalence'].values ) 
            #print('log odds before regression', log_odds) 
            print("beta", self.beta_reg)
            print(self.beta_reg.shape, Y.shape)
            log_odds +=  Y.dot(self.beta_reg) # - log_odds_prev
            print('log odds after regression', log_odds)
            #print('log odds after regression', log_odds)

        if self.separate_asymptomatic:
            index_sympt = np.where(data['symptomatic']==1)[0]
            ### Start by updating the symptomatic columns
            if self.use_prevalence_ili == False:
                X  = data['symptomatic'].values
                log_odds += self.update_logs_bernouilli(X, 'symptomatic')

            
        for k in self.test_groups:
              index = np.where(data['label_t']==k)
              log_odds[index] += self.update_logs_bernouilli(T[index], 'T_' + k)
        for n in self.list_symptoms:
            X  = data[n].iloc[index_sympt].values
            log_odds[index_sympt] += self.update_logs_bernouilli(X, n) 
            #print('log odds',n, log_odds)
        for n in self.list_symptoms_quant:
            #### We sample a bunch as we parametrize it by beta dist
            X  = data[n].iloc[index_sympt].fillna(0).values
            log_odds[index_sympt] += self.update_logs_prior(X, n)

        imputed_labels = np.divide(np.ones(N),
                         np.ones(N) + np.exp(log_odds))
        if self.stochastic:
            imputed_labels = np.random.binomial(n=1, p=imputed_labels)
        return imputed_labels

    def maximization(self, data, T, imputed_labels, agnostic_init=False):
        ''' Maximization of the parameters for the model, that is the $beta$ and coefficients
        for the symptoms
        '''
        N,_ = data.shape
        lab_t = data['label_t']
        if self.separate_asymptomatic:
            index_sympt = np.where(data['symptomatic']==1)[0]
            X = data['symptomatic'].values
            if self.update_symptomatic:
                add_alpha1, add_beta1 = self.add_updates2beta_prior(imputed_labels, X)
                add_alpha0, add_beta0 = self.add_updates2beta_prior(1. - imputed_labels, X)
                self.update_bernouilli('symptomatic_1')
                self.update_bernouilli('symptomatic_0')

            if (self.update_prior_symptoms) or (not agnostic_init):
              self.update_beta('symptomatic_1', add_alpha1, add_beta1)
              self.update_beta('symptomatic_0', add_alpha0, add_beta0)
                
        else:
            index_sympt = np.arange(data.shape[0])

        if self.update_sensitivity:
            ### Start by updating the sensitivity, which is the same for everyone
            add_alpha0, add_beta0 = self.add_updates2beta_prior(1. - imputed_labels, T)
            self.update_beta('T_0', add_alpha0, add_beta0)
            for k in self.specificity.keys():
              index = np.where(lab_t == k)[0]
              add_alpha1, add_beta1 = self.add_updates2beta_prior(imputed_labels[index], T[index])
              #add_alpha0, add_beta0 = self.add_updates2beta_prior(1. - imputed_labels[index], T[index])
              self.update_beta('T_' + k + '_1', add_alpha1, add_beta1)
              #self.update_beta('T_' + k + '_0', add_alpha0, add_beta0)
            self.update_bernouilli('T_' + k  + '_1')
            self.update_bernouilli('T_' + k + '_0')
        for n in self.list_symptoms:
            X = data[n].iloc[index_sympt].values
            temp_lab =  imputed_labels[index_sympt]
            add_alpha1, add_beta1 = self.add_updates2beta_prior(temp_lab, X)
            add_alpha0, add_beta0 = self.add_updates2beta_prior(1. - temp_lab, X)
            if (self.update_prior_symptoms) or (not agnostic_init):
              self.update_beta(n + '_1', add_alpha1, add_beta1)
              self.update_beta(n + '_0', add_alpha0, add_beta0)
              self.update_bernouilli(n + '_1')
              self.update_bernouilli(n + '_0')
        for n in self.list_symptoms_quant:
            X = data[n].iloc[index_sympt].values
            temp_lab =  imputed_labels[index_sympt]
            self.update_quant_prior(X, temp_lab, n + '_1')
            self.update_quant_prior(X, 1 - temp_lab, n + '_0')
        
        Y =  - np.log(imputed_labels) + np.log(1.0 - imputed_labels)  ### we are doing a linear regression of the odds-ratio
        Y = Y - (- np.log(data['prevalence'].values) +  np.log(1.0 - data['prevalence'].values ))
        self.update_mvn('reg', Y,  data)

        #self.update_mvn_known_Sig('beta', data, Y)
        self.beta_reg = self.params['mu_reg']
   


    def fit(self, data, T, priorD):
        """ Run the EM for several steps
        """
        assert self.method_missing_T in [
            'truncated', 'latent-DT', 'latent-DT-newprior']

        #### Step `. initialize values
        N = len(T)
        T_missing = np.isnan(T)
        
        T_not_missing = ~T_missing
        lab_t = data['label_t']
        
        imputed_labels = copy.deepcopy(priorD).astype('float')  ### initialize at image?
        print("There are %i nan imputed labels before starting"%np.sum(np.isnan(imputed_labels)))
        self.maximization(
                data, imputed_labels, imputed_labels, 
                agnostic_init=self.completely_agnostic_prior)

        it_em = 0
        converged = False
        list_coef = ['p_T_0'] + ['p_T_' + k + '_1'
                            for k in self.test_groups]
        list_coef += [ 'p_' +  n + '_' + e
                            for e in ['0','1'] for n in self.list_symptoms + ['symptomatic']]
        while not converged:
            print("Fit: Iteration %i..." % it_em)

            imputed_labels = self.expectation(data, T)
                #print("Done imputing")

            imputed_labels[np.where(imputed_labels>0.99999)] = 0.99999
            imputed_labels[np.where(imputed_labels<0.00001)] = 0.00001
            print("There are %i nan imputed labels"%np.sum(np.isnan(imputed_labels)))
            old_params = np.array([self.params[k] for k in list_coef] )#+ list(self.beta_reg))
            old_reg  = self.beta_reg
            self.maximization(data, T, imputed_labels, not self.update_prior_symptoms)
            new_params = np.array([self.params[k] for k in list_coef]) #+ list(self.beta_reg))
            diff = np.sqrt(np.sum((new_params - old_params)**2)) *1.0/len(old_params) + np.sqrt(np.sum((old_reg  - self.beta_reg)**2))/np.sqrt(0.01 + np.sum(old_reg**2))
            it_em += 1
            converged= (it_em>self.EM_steps) or (diff < self.convergence_threshold)
            print("Diff: %1.4f "%diff)
        self.convergence = it_em
        return(imputed_labels)

    def save(self, filename):
        dict_v = {'params': self.params, 'hyperparams': self.hyperparams} ### we just need the prior parameters for our models
        pickle.dump(dict_v, open(filename, 'wb'))

    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.params = dict_v['params']
        self.hyperparams = dict_v['hyperparams']

    def log_odds_given_X(self, data):
        ''' Produces a log_odds posterior credible distribution for the immunity score given context
        + questionnaire
        Just have to unfold the model progressively: sample from the different distributions
        the different log ratios
        '''
        #print("Howdy")
        Y = data[self.list_context_variables].values
        index_sympt = np.where(data['symptomatic']==1)[0]
        N, _ = Y.shape
        log_odds = np.zeros((N, self.B))
        if self.use_prevalence:
          log_odds_prev = np.log(data['prevalence'].values) - np.log(1.0 - data['prevalence'].values )
          log_odds = np.hstack([-log_odds_prev.reshape([-1,1])]* self.B)
        elif self.use_prevalence_ili:
            log_odds_prev = np.multiply(data['symptomatic'].values, 
                                        np.log(data['p01'].values)- np.log(1.0 - data['p11'].values )
                            ) + np.multiply(1.0 - data['symptomatic'].values, 
                                        np.log(data['p00'].values)- np.log(1.0 - data['p10'].values )
                            )
            log_odds = np.hstack([-log_odds_prev.reshape([-1,1])]* self.B)
        else:
          #print("here is x")
          #print("prev", data['prevalence'].values, self.variability_prev)
          def temp_function(x): 
              temp = np.random.normal(x, np.sqrt(self.variability_prev),1)
              while temp < 0 or temp>1:
                  temp = np.random.normal(x, np.sqrt(self.variability_prev),1)
              #a,b = moment_matching_beta(x, self.variability_prev)
              #print( ["a, b", a,b])
              #return(np.random.beta(a,b, self.B).T)
              return temp
          log_odds_prev  = np.apply_along_axis(lambda x: temp_function(x),
                                     1, data['prevalence'].values.reshape([-1,1]))
          #print(log_odds_prev)
          log_odds_prev = - np.log(log_odds_prev) + np.log(1.0 - log_odds_prev ) # +\
          X = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                          data[self.list_context_variables].values.astype('float')])
          log_odds = np.apply_along_axis(lambda x: x.dot(np.random.multivariate_normal(self.params['mu_reg'],
                                                            sc.stats.invgamma.rvs(self.params['a_reg'],self.params['b_reg']) * np.linalg.inv(self.params['Lambda_reg']),
                                                            self.B).T),
                    1,X)
          log_odds += log_odds_prev #np.hstack([log_odds_prev.reshape([-1,1])]* self.B)
        if self.separate_asymptomatic:
            n = 'symptomatic'
            X = data[n].values
            if self.use_prevalence_ili == False:
              x = np.reshape(np.random.beta(self.params['alpha_'+ n + '_1'],
                                self.params['beta_'+ n + '_1'],
                                self.B * N),
                            (N, self.B))### Sample from the distribution?
              y = np.reshape(np.random.beta(self.params['alpha_'+ n + '_0'],
                                self.params['beta_'+ n + '_0'],
                                self.B * N),
                            (N, self.B))### Sample from the distribution?
              log_odds +=  self.update_logs_samplebernouilli(X, y) \
                          - self.update_logs_samplebernouilli(X, x)

        else:
            index_sympt =np.arange(N)
        if len(index_sympt)> 0:
          for n in self.list_symptoms:
                X = data[n].iloc[index_sympt]
                NN = len(index_sympt)
                x = np.reshape(np.random.beta(self.params['alpha_'+ n + '_1'],
                                  self.params['beta_'+ n + '_1'],
                                  self.B * NN),
                              (NN, self.B))### Sample from the distribution?
                y = np.reshape(np.random.beta(self.params['alpha_'+ n + '_0'],
                                  self.params['beta_'+ n + '_0'],
                                  self.B * NN),
                              (NN, self.B))### Sample from the distribution?
                log_odds[index_sympt]+= self.update_logs_samplebernouilli(X, y) \
                            - self.update_logs_samplebernouilli(X, x)
          for n in self.list_symptoms_quant:
              X = data[n].iloc[index_sympt]
              NN = len(index_sympt)
              sample0 = self.sample_quant(n + '_0', NN)
              sample1 = self.sample_quant(n + '_1', NN)
              log_odds[index_sympt]+= self.update_logs_samplequant(X, sample0)\
                                - self.update_logs_samplequant(X, sample1)          
        return(log_odds)
    
    def posterior_given_X(self, data):
      log_odds = self.log_odds_given_X(data)
      return np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
      
    def log_odds_given_XandT(self, data, T):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        
        N, _ = Y.shape
        T_missing = np.isnan(T) #| (data.test == 0)
        T_not_missing = ~T_missing
        if self.separate_asymptomatic:
            index_sympt = np.where(data['symptomatic']==1)[0]
        else:
            index_sympt =np.arange(N)
        log_odds = self.log_odds_given_X(data)
        n = 'T'
        for k in self.test_groups:  
            index = np.where((data['label_t']==k))[0]
            index_missing = np.where((data['label_t']==k) & ( T_missing))[0]
            index_not_missing = np.where((data['label_t']==k) & ( T_not_missing))[0]
            NN = len(index)
            
            X = T[index_not_missing]
            print(k,NN, len(index_not_missing), len(index_missing), X )
            x = np.random.beta(self.hyperparams['alpha_'+ n + '_' + k +  '_1'],
                              self.hyperparams['beta_'+ n + '_'  + k + '_1'],
                              self.B * NN)
            x[np.where(x>0.9999999)] = 0.9999999
            x[np.where(x<0.0000001)] = 0.0000001
            x = np.reshape(x, (NN, self.B))### Sample from the distribution?
            y = np.random.beta(self.hyperparams['alpha_T_0'],
                              self.hyperparams['beta_T_0'],
                              self.B * NN)
            y[np.where(y>0.99999)] = 0.99999
            y[np.where(y<0.0000001)] = 0.0000001
            y = np.reshape(y,(NN, self.B))
            log_odds[index_not_missing] = log_odds[index_not_missing] + self.update_logs_samplebernouilli(X, y[:len(index_not_missing),:]) \
                            - self.update_logs_samplebernouilli(X, x[:len(index_not_missing),:])

        return(log_odds)
    
    
    def posterior_given_XandT(self, data, T):
      log_odds = self.log_odds_given_XandT(data, T)
      return np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))


algo = EMClassifier3(EM_steps=200, 
                    beta=np.zeros((1+ len(CONTEXT_INFO))),
                    list_symptoms=SYMPTOMS,
                    list_symptoms_quant=SYMPTOMS_QUANT,
                    list_context_variables= CONTEXT_INFO,
                    specificity_priors=SPECIFICITY_PRIORS,
                    sensitivity_priors=SENSITIVITY_PRIORS,
                    group_tests = group_tests,
                    hyperparams_dist = hyperparams,
                    params_dist = hyperparams,
                    separate_asymptomatic=True,
                    use_prevalence=False,
                    use_prevalence_ili=False,
                    update_sensitivity= False,
                    update_symptomatic=True,
                    stochastic=False,
                    convergence_threshold = 1e-5,
                    completely_agnostic_prior=False,
                    update_prior_symptoms=True,
                    B=1000
                    )
algo.load('weights.pkl')
algo.params['p_symptomatic_1'] = 1.0 - 0.308
params_asymptomatics  = moment_matching_beta(1.0 - 0.308,
                                      (1.0/(2*1.96) *(0.538 - 0.077))**2) 
algo.params['alpha_symptomatic_1'] = params_asymptomatics[0]
algo.params['beta_symptomatic_1'] = params_asymptomatics[1]
algo.hyperparams['alpha_T_0'] = 1.0
algo.hyperparams['beta_T_0'] = 1000000
algo.hyperparams['p_T_0'] = 1.0/1000001.0
algo.params['alpha_T_0'] = 1.0
algo.params['beta_T_0'] = 1000000.0
algo.params['p_T_0'] = 1.0/1000001.0


def testMethod(achyJointsMuscles, runny_nose,
           cough,fatigue,fever,headache,lossTaste,lossSmell,
           shortnessOfBreath, soreThroat, stomachUpsetDiarrhoea,
           householdIllness,numberInHousehold, numberIllInHousehold, percentage_householdIllness,
           fever_severity, cough_severity, howShortOfBreath,high_risk_exposure_occupation, high_risk_interactions, 
           date_test, type_test, test, symptom_onset, symptomatic, country, region, prev,
           pct_ili, pct_cli, pct_tested, sd_prev, sd_ili, sd_cli, sd_tested): ##get number of bins passed by R Shiny server
  d = date_test
  date_test= datetime(d.year, d.month, d.day)
  if np.isnan(symptom_onset): days_to_subtract = 0
  else: days_to_subtract = symptom_onset

  if date_test < datetime.strptime("2020-05-07", "%Y-%m-%d"):
      date_test = datetime.strptime("2020-05-08", "%Y-%m-%d")
  if date_test > datetime.strptime("2020-09-24", "%Y-%m-%d"):
      date_test = datetime.strptime("2020-09-24", "%Y-%m-%d")
  date_test = date_test - timedelta(days=days_to_subtract)

  train_data = pd.DataFrame.from_dict({
    'myalgia': [int(achyJointsMuscles)],
    'runny_nose': [int(runny_nose)],
    'cough': [int(cough)],
    'fatigue': [int(fatigue)],
    'fever': [int(fever)],
    'headache': [int(headache)],
    'loss_of_taste': [int(lossTaste)],
    'loss_of_smell': [int(lossSmell)],
    'sob':[int(shortnessOfBreath)], 
    'sore_throat': [int(soreThroat)], 
    'diarrhea': [int(stomachUpsetDiarrhoea)],
    'householdIllness': [int(householdIllness)],
    'numberInHousehold':[ int(numberInHousehold)], 
    'percentage_householdIllness': [percentage_householdIllness],
    'fever_severity': [int(fever_severity)], 
    'cough_severity': [int(fever_severity)], 
    'howShortOfBreath':[int(howShortOfBreath)], 
    'high_risk_exposure_occupation': [float(high_risk_exposure_occupation)],
    'high_risk_interactions': [float(high_risk_interactions)],
    'numberIllInHousehold': [int(numberIllInHousehold)],
    'symptom_onset' : [int(symptom_onset)],
    'test':[int(test)],
    'symptomatic':[int(symptomatic)],
    'p11': [(1-0.16) * prev],  ### discount for asymptomatic
    'p10': [ 0.16 * prev],  ### discount for asymptomatic
    'p01': [pct_ili * 0.01],
  },  orient='columns')
  
  #### The local prevalence is contigent on the sampling frame
  #### if the person is asymptomatic,it's the general prevalence in the overall population
  #### if the person has been tested, it's the prevalence among tested
  #### if the person has symptoms, its the prevalence among symptomatic (which we approximate by the n of peopke tested)
  if int(symptomatic) == 1: 
       #### prevalence among symptomatics
       a, b = moment_matching_beta(pct_cli * 0.01, (sd_cli * 0.01)**2)
       r_cli = np.random.beta(a,b, 1000)
       a, b = moment_matching_beta(pct_ili * 0.01, (sd_ili * 0.01)**2)
       r_ili = np.random.beta(a,b, 1000)
       
       train_data['prevalence'] = pct_cli/(pct_cli+pct_ili)
       var_prev = np.var(r_cli/(r_cli+r_ili))
  else: 
     ### prevalence among tested (a little different than general)
     ### first correct pct_tested
     if test < 2 and pct_tested >  0: 
         a, b = moment_matching_beta(pct_cli * 0.01, (sd_cli * 0.01)**2)
         r_prev = np.random.beta(a,b, 1000)
         a,b = moment_matching_beta(pct_tested * 0.01, (sd_tested * 0.01)**2)
         r_pct_tested = np.random.beta(a,b, 1000)
         ratio = np.divide(r_prev,r_pct_tested)
         ratio = ratio[ratio<1]
         train_data['prevalence'] =  np.median(ratio)
         var_prev =  np.var(ratio)
     else:
         train_data['prevalence'] = prev
         var_prev  = sd_prev**2
         
  train_data['p00'] = 1.0 - train_data['p01'] - train_data['p10'] - train_data['p11']
  algo.variability_prev = var_prev
  T = np.array([test]).astype('int')
  if test == 2:
    train_data['label_t'] = "missing"
  if symptomatic == 0:
    train_data['label_t'] = type_test + "_asymptomatic"
  if np.isnan(symptom_onset):
    train_data['label_t'] = type_test + "_undefined"
  else:
    if type_test  == "pcr":
      if int(symptom_onset) < 18: train_data['label_t'] = "pcr_"+str(int(symptom_onset))
      else: train_data['label_t'] = "pcr_undefined"
    else:
      if int(symptom_onset) < 6: train_data['label_t'] = 'rapid_1-5 days'
      elif  int(symptom_onset) < 11: train_data['label_t'] = 'rapid_6-10 days'
      elif  int(symptom_onset) < 16:train_data['label_t'] = 'rapid_11-15 days'
      elif  int(symptom_onset) < 21:train_data['label_t'] = 'rapid_16-20 days'
      else: train_data['label_t'] = 'rapid_21+ days'

  if test < 2:
      #print(['test', test, T])
      post = algo.posterior_given_XandT(train_data, T-0.0001)
      #print(post)
  else:
      post = algo.posterior_given_X(train_data)
  return post