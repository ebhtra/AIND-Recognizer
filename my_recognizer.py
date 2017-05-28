import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    #build a probability dict for each test word
    for X, L in test_set.get_all_Xlengths().values():
        probDict = {}
        # try each test word in each trained word's model to see how it fits
        for wordkey in models:
            try:
                score = models[wordkey].score(X, L)
                probDict[wordkey] = score
            except:
                # need following line to pass automated test, which requires
                #   a probDict for every test word, even if it can't be scored:
                probDict[wordkey] = -float('inf')
 #               print("couldn't score word ", wordkey)
                continue
            
        probabilities.append(probDict)
    # return the best fit for each test word    
    guesses = [max(d, key=lambda x: d[x]) for d in probabilities]
    # not clear yet why we need to have the probabilities list, or return it
    return probabilities, guesses
    
