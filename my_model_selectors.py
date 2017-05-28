#import math
#import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        
        # keep track of best number of states, to return a base model with that
        best_n = self.min_n_components
        best_score = float('inf')
        
        # loop through number of hidden states to find best BIC score
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                # train a model with current n_states
                model = GaussianHMM(n_components=n_states,
                                    n_iter=1000).fit(self.X, self.lengths)
                score = model.score(self.X, self.lengths)
                # find BIC score
                # formula is from website listed above in docstring
                # 'p' for formula uses docstring quoted below
                n_features = len(self.sequences[0][0])
                n_covar_params = n_states * n_features  #shape of hmm diag. covar matrix (see below)
                n_means_params = n_states * n_features  #shape of hmm means matrix (see below)
                n_start_params = n_states             #length of prior distribution
                n_transmat_params = n_states * n_states
                n_params = n_covar_params + n_means_params \
                         + n_start_params + n_transmat_params
                n_datapoints = len(self.X)  # Helpful udacity forum discussion
                    # for this: https://discussions.udacity.com/t/number-of-data-points-bic-calculation/235294/2
                    
                bic_score = -2 * score + n_params * np.log(n_datapoints)
                
                if bic_score < best_score:
                    best_score = bic_score
                    best_n = n_states
                
            except:
                continue
#                print('unable to calculate BIC score for n_states = ', n_states)
                
        return self.base_model(best_n)
        """ 
        --> n_params was calculated using the following part of 
            the docstring for GaussianHMM class from hmm.py (imported
            into this project) by Ron Weiss et al.
            -----------------------------------------------------
        "transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

        startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

        means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

        covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_features, n_features)                if "tied",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"  "
        """

                

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        
        # keep track of best number of states, to return a base model with that
        best_score = -float('inf')
        best_n = self.min_n_components

        # main loop, over all n_states, to find where current word is
        #    most likely, compared to avg of all other words
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                # train a model with current n_states
                model = GaussianHMM(n_components=n_states,
                                    n_iter=1000).fit(self.X, self.lengths)
                # log-likelihood for the training word
                score = model.score(self.X, self.lengths)
                # log-likelihood for all other words
                otherScores = []
                for word in self.hwords:
                    wordX, wordlengths = self.hwords[word]
                    otherScores.append(model.score(wordX, wordlengths))
                #calculate the DIC score
                dic_score = score - sum(otherScores) / len(otherScores)
                if dic_score > best_score:
                    best_score = dic_score
                    best_n = n_states
            except:
                continue
#                print('failed to get a model score for n_states = ', n_states,
#                      ' for word: ', self.this_word)
        
        return self.base_model(best_n)    



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
       
        # implement model selection using CV
        
        if len(self.sequences) < 2: # can't cross-validate fewer than 2 sets
            return None
        # get the train/test split indexes for each fold
        if len(self.sequences) == 2:
            splits = KFold(n_splits=2).split(self.sequences)
        else:
            splits = KFold(n_splits=3).split(self.sequences)
        
        # train and test each split for each number of hidden states
        # keep track of best number of states, to return a base model with that
        best_n = self.min_n_components
        best_score = -float('inf')
        
        # format the splits outside of the main loop, since they're in a generator
        trainers = []
        testers = []
        for split in splits:
            # use the imported asl_utils method to format the data for hmmlearn
            trainers.append(combine_sequences(split[0], self.sequences))
            testers.append(combine_sequences(split[1], self.sequences))
            
        # main loop, through number of hidden states:

        for n_states in range(self.min_n_components, self.max_n_components+1):
            logScores = []
            for fold in range(len(trainers)):
                trainX, trainLengths = trainers[fold]
                testX, testLengths = testers[fold]
                
                try:
                    # train the model with those splits and current n_states
                    model = GaussianHMM(n_components=n_states,
                                        n_iter=1000).fit(trainX, trainLengths)
                    # add test score to running total
                    score = model.score(testX, testLengths)
                    logScores.append(score)
                    
                except:
                    continue
#                    print("model didn't complete training through n_states")
                    
                
            #average the scores of the splits to see if new best
            if logScores:
                avg = sum(logScores) / len(logScores)
                
                if avg > best_score:
                    best_score = avg
                    best_n = n_states
            
               
        return self.base_model(best_n)    
        
    
