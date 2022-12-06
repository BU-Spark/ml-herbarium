# Prediction Functions for the Project

# Extra colors for fancy printing
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_prediction(all_matches,comparison_file,word_log_dic,syn_pure, minimum_similarity = 0):
    """
    Given a dictionary of matching information, a comparison file containing ground truth data, a dictionary of word
    log information, and a list of synonyms, this function prints the predicted strings and their corresponding
    ground truth data and similarity score.
    
    Parameters:
    all_matches (dict): A dictionary containing the matching information for multiple files, with keys the
                        same name as the input files and values as the DataFrame with matching information.
    comparison_file (dict): A dictionary containing the ground truth data for the input files, with the file
                            names as the keys and the strings as the values.
    word_log_dic (dict): A dictionary containing information about the words in the input files, with the word
                         as the key and the image number as the value.
    syn_pure (list): A list of synonyms to use when comparing the predicted strings to the ground truth data.
    minimum_similarity (float): The minimum similarity required for a prediction to be printed (default is 0).
    
    Returns:
    None, but prints the predicted strings and their corresponding ground truth data and similarity score.
    Green text indicates a correct prediction, red text indicates an incorrect prediction, and cyan text
    indicates a synonym of the ground truth data.
    """

    # Print the predictions for the 
    for k,v in all_matches.items():    
        # predicted text for a given image
        prediction_and_imagenumber = list(zip(v[k+'_Corpus'], v.right_index))
        syn_matches = 0
        correct = 0
        guessed = 0
        count = 0
        if k == 'Taxon' or k == 'Countries': # this is just for the two ground truth files we have
            print(color.BOLD+'Evaluation for '+str(k)+color.END)
            for idx,(prediction,image_number) in enumerate(prediction_and_imagenumber):
                try:
                    if v.loc[v['right_index'] == idx, 'similarity'].iloc[0]>minimum_similarity:
                        try:
                            syn = syn_pure[prediction.lower()]
                        except KeyError as e:
                            syn = []
                        try:
                            image = word_log_dic[image_number]
                            gt = comparison_file[k][image]
                            if gt == prediction:
                                correct +=1
                                print(color.GREEN+gt+"||",prediction,'||',image,'||'+str(v.loc[v['right_index'] == idx, 'similarity'].iloc[0])+color.END)
                            elif syn:
                                # print(syn)
                                if gt.lower() == syn.lower():
                                    syn_matches +=1
                                    correct +=1
                                    print(color.DARKCYAN+gt+"||",prediction,'||',image,'||'+str(v.loc[v['right_index'] == idx, 'similarity'].iloc[0])+color.END)
                                else:
                                    print(color.RED+gt+"||"+prediction+'||'+str(image)+'||'+str(v.loc[v['right_index'] == idx, 'similarity'].iloc[0])+color.END)
                            else:
                                print(color.RED+gt+"||"+prediction+'||'+str(image)+'||'+str(v.loc[v['right_index'] == idx, 'similarity'].iloc[0])+color.END)
                        except (KeyError,IndexError) as e:
                            # if e == KeyError:
                            #     pass
                            #     # print('KeyError')
                            #     # print(v.iloc[idx])
                            # else:
                            #     pass
                            #     # print('IndexError\nThere is likely no predicted value for this image.')
                            #     # print(v.iloc[idx])
                            print("Ground Truth Not Found for:",word_log_dic[image_number])
                        guessed+=1
                except:
                    pass
            print(color.BOLD+"Accuracy on Predicted:"+str(correct/guessed)+color.END)
            print(color.BOLD+"Synonym Matches:"+str(syn_matches/len(v))+color.END)
            print(color.BOLD+"Total accuracy: "+str(correct/len(v))+color.END)
            print(color.BOLD+"Total Guessed:"+str(guessed)+color.END)
            print(color.BOLD+"Percentage Guessed:"+str(guessed/len(v))+color.END)
            print('\n\n********************************\n\n')


def check_accuracy(all_matches,syn_pure,word_log_dic,comparison_file,min_simil):
    """
    Given a dictionary of matching information, a list of synonyms, a dictionary of word log information,
    and a comparison file containing ground truth data, this function returns several measures of accuracy
    for the matches.
    
    Parameters:
    all_matches (dict): A dictionary containing the matching information for multiple files, with keys the
                        same name as the input files and values as the DataFrame with matching information.
    syn_pure (list): A list of synonyms to use when comparing the predicted strings to the ground truth data.
    word_log_dic (dict): A dictionary containing information about the words in the input files, with the word
                         as the key and the image number as the value.
    comparison_file (dict): A dictionary containing the ground truth data for the input files, with the file
                            names as the keys and the strings as the values.
    min_simil (float): The minimum similarity required for a match to be considered correct (default is 0.001).
    
    Returns:
    tuple: A tuple containing the following values:
           - acc_pred (list): The accuracy of the predicted strings (correctly predicted / total predicted).
           - total_acc (list): The total accuracy of the matches (correctly predicted / total matches).
           - synonym_match (list): The proportion of correct matches that were synonyms of the ground truth data.
           - types (list): The names of the input files used in the matching process.
           - total_guessed (list): The total number of guesses.
    """
    acc_pred = []
    total_acc = []
    types = []
    synonym_match = []
    total_guessed = []
    for k,v in all_matches.items():    
        # predicted text for a given image
        prediction_and_imagenumber = list(zip(v[k+'_Corpus'], v.right_index))
        syn_matches = 0
        correct = 0
        guessed = 0
        num_predicted = 0
        if k == 'Taxon' or k == 'Countries': # this is just for the two ground truth files we have
            for idx,(prediction,image_number) in enumerate(prediction_and_imagenumber):
                if v.loc[v['right_index'] == idx, 'similarity'].iloc[0]>min_simil:
                    try:
                        syn = syn_pure[prediction.lower()]
                    except KeyError as e:
                        syn = []
                    try:
                        image = word_log_dic[image_number]
                        gt = comparison_file[k][image]
                        if gt == prediction:
                            correct +=1
                        elif syn:
                            if gt.lower() == syn.lower():
                                syn_matches +=1
                                correct +=1
                        else:
                            pass
                    except (KeyError,IndexError) as e:
                        print("Ground Truth Not Found for:",word_log_dic[image_number])
                    guessed+=1

            acc_pred.append(correct/guessed)
            total_acc.append(correct/len(v))
            synonym_match.append(syn_matches/correct)
            types.append(k)
            total_guessed.append(guessed)
    return acc_pred,total_acc,synonym_match,types,total_guessed