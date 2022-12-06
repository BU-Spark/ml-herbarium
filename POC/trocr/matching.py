import multiprocessing
import pandas as pd
from functools import partial
from itertools import repeat
from string_grouper import match_strings

# All matching functions used in the project

def matches_above_x(df,x = .75):
    # Takes in a dataframe with similarity scores and returns a dataframe with only matches above a certain similarity
    return df.loc[df['similarity']>=x]

def match(main,comparison_file,minimum_similarity=.001):
     # Function takes a main file containing strings, a comparison file to match against main,
     #  and a minimum similarity confidence level. Returns a list of matches based on similarity.

    if not isinstance(comparison_file, pd.Series):
        comparison_file = pd.Series(comparison_file)
        
    matches = match_strings(main,comparison_file,n_blocks = 'guess',min_similarity = minimum_similarity,max_n_matches = 1)

    return matches

def highest_score_per_image(df,labels):
    # Takes in a dataframe with similarity scores and returns a dataframe with only the highest score per image

    # Getting the highest score for each individual image 
    exclude = ['Don Williams','Don William']
    index_to_labels = df.copy()
    for a in index_to_labels.right_index.unique():
        index_to_labels.loc[index_to_labels['right_index'] == a, 'right_index'] = labels[a]
    #drop all rows whose column value in Predictions is in exclude
    index_to_labels = index_to_labels[~index_to_labels['Predictions'].isin(exclude)]

    unique_labels = index_to_labels.loc[index_to_labels.groupby('right_index')['similarity'].idxmax()]


    return unique_labels

def pooled_match(comparison_file,labels, minimum_similarity = .001,**kwargs):
    # Take in any number of files containing strings to match against and returns a dictionary
    # with keys the same name as input and values as the dataframe with matching information
   
    corpus_list = []
    corpus_name = []
    
    for k,v in kwargs.items():
        # Convert to series (string-grouper requires this type), will work if input is list, array, or series
        if not isinstance(v, pd.Series):
            v = pd.Series(v)
        corpus_list.append(v)
        corpus_name.append(k)

    if not isinstance(comparison_file, pd.Series):
        comparison_file = pd.Series(comparison_file)

    func = partial(match, comparison_file = comparison_file,  minimum_similarity = minimum_similarity)
    pool = multiprocessing.Pool()
   
    result_dic = {}
    for i,result in enumerate(pool.map(func,corpus_list)):
        result.columns.values[1] = corpus_name[i]+'_Corpus'
        result.columns.values[3] = "Predictions"
        result = result.drop('left_index', axis=1)
        result = highest_score_per_image(result,labels)
        result_dic[corpus_name[i]] = result
   
    return result_dic

def pooled_match2(comparison_file,labels, minimum_similarity = .001,**kwargs):
    # Take in any number of files containing strings to match against and return a dictionary
    # with keys the same name as input and values as the dataframe with matching information
   
    corpus_list = []
    corpus_name = []
    
    for k,v in kwargs.items():
        # Convert to series (string-grouper requires this type), will work if input is list, array, or series
        if not isinstance(v, pd.Series):
            v = pd.Series(v)
        corpus_list.append(v)
        corpus_name.append(k)
    
    if not isinstance(comparison_file, pd.Series):
        comparison_file = pd.Series(comparison_file)

    # func = partial(match, comparison_file = comparison_file,  minimum_similarity = minimum_similarity)
    pool = multiprocessing.Pool()
   
    result_dic = {}
    for i,result in enumerate(pool.starmap(match, zip(corpus_list, repeat(comparison_file),repeat(minimum_similarity)))):
        result.columns.values[1] = corpus_name[i]+'_Corpus'
        result.columns.values[3] = "Predictions"
        result = result.drop('left_index', axis=1)
        result = highest_score_per_image(result,labels)
        result_dic[corpus_name[i]] = result
   
    return result_dic


 # Function to add in the bigram index as a column on the dataframe
def bigram_indices(row):
    """
    Given a row of data with columns 'Transcription' and 'Bigrams', this function
    returns a list of tuples containing the indices of the words in the 'Transcription'
    column that correspond to the bigrams in the 'Bigrams' column.
    
    Parameters:
    row (pandas.Series): A row of data with columns 'Transcription' and 'Bigrams'.
    
    Returns:
    list: A list of tuples, each containing the indices of the words in the 'Transcription'
    column that correspond to the bigrams in the 'Bigrams' column.
    
    Example:
    row = pd.Series({'Transcription': ['Herbier Museum.', 'Paris Cryptogamie.', 'PC0693452', '0.'],
    'Bigrams': ['Herbier Museum.','Museum. Paris','Paris Cryptogamie.','Cryptogamie. PC0693452','PC0693452 0.']})
    bigram_indices(row)
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    """
    list_of_transcriptions =  [list(x.split(' ')) for x in row['Transcription']]

    bigram_idx = []
    for word in row['Bigrams']:
        strings = word.split(' ')
        for i,x in enumerate(list_of_transcriptions):
            if strings[0] in x:
                first_idx = i
            if strings[1] in x:
                second_idx = i
        bigram_idx.append((first_idx, second_idx))
    return bigram_idx
