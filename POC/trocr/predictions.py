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

    # Takes in the matches dictionary and checks results against ground truth
    # Defaults to 0 similarity(returning values for all matches), but can be changed to any value between 0 and 1

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
            print(color.BOLD+"Accuracy on Predicted:"+str(correct/guessed)+color.END)
            print(color.BOLD+"Synonym Matches:"+str(syn_matches/len(v))+color.END)
            print(color.BOLD+"Total accuracy: "+str(correct/len(v))+color.END)
            print(color.BOLD+"Total Guessed:"+str(guessed)+color.END)
            print(color.BOLD+"Percentage Guessed:"+str(guessed/len(v))+color.END)
            print('\n\n********************************\n\n')


def check_accuracy(all_matches,syn_pure,word_log_dic,comparison_file,min_simil):
    acc_pred = []
    total_acc = []
    types = []
    s_match = []
    co = []
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
            s_match.append(syn_matches/correct)
            types.append(k)
            co.append(guessed)
    return acc_pred,total_acc,s_match,types,co