import cv2
import matplotlib.pyplot as plt
# Functions for Displaying Predictions


def get_color(similarity):
    """
    Given a similarity score, this function returns a color corresponding to the score.
    
    Parameters:
    similarity (float): A similarity score between 0 and 1.
    
    Returns:
    tuple: A tuple representing a color, with the format (R, G, B), where R, G, and B are integers between 0 and 255.
    """

    # if similarity is greater than .8, return green color
    if similarity > .8:
        return (0,255,0)
    # if similarity is greater than .5, return yellow color
    elif similarity > .5:
        return (0,255,255)
    # if similarity is greater than .3, return orange color
    elif similarity > .3:
        return (0,165,255)
    # if similarity is less than .3, return red color
    else:
        return (0,0,255)


def all_boxes(df,label):
    """
    Given a dataframe and a label, this function displays the image associated with the label, along with bounding boxes for all the text in the image.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing information about the images and bounding boxes.
    label: The label for the image to be displayed.
    
    Returns:
    None: This function does not return anything. It simply displays the image with bounding boxes.
    """

    image_path = df.loc[df['Labels'] == label, 'Image_Path'].iloc[0]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(50, 50))
    ax.imshow(img)
    for idx,box in enumerate(df.loc[df['Labels'] == label,'Bounding_Boxes'].iloc[0]):
        rect = cv2.boundingRect(box)  # returns (x,y,w,h) of the rect
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)
        cv2.putText(img, str(idx), (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    plt.imshow(img)
    plt.show()



def display_image(df,all_matches,Taxon_truth,Geography_truth,word_log_dic,label):
    """
    Given a dataframe, all_matches, taxon and geography ground truth, word_log_dic and a label, this function displays the image associated with the label, along with bounding boxes for all the text in the image.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing information about the images and bounding boxes.
    label: The label for the image to be displayed.
    Taxon_truth (dict): A dictionary containing the ground truth for the taxon.
    Geography_truth (dict): A dictionary containing the ground truth for the geography.
    word_log_dic (dict): A dictionary containing the label to image name mapping.
    
    Returns:
    None: This function does not return anything. It simply displays the image with bounding boxes and associated transcription and similarity score.
    """
    
    THICKNESS = 2
    # get the image path for the specified label
    print('Taxon Predicted GT: ',df.loc[df['Labels'] == label, 'Taxon_Prediction'].iloc[0])
    print('Taxon GT: '+Taxon_truth[word_log_dic[label]])
    print('Countries GT: '+Geography_truth[word_log_dic[label]])
    print('Countries Predicted GT: ',df.loc[df['Labels'] == label, 'Countries_Prediction'].iloc[0])
    image_path = df.loc[df['Labels'] == label, 'Image_Path'].iloc[0]
    img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(50, 50))
    ax.imshow(img)
    for k,_ in all_matches.items():
        # Prevents double printing the same text
        found1 = False
        if k == 'Taxon' or k == 'Countries': # probably want to just focus on taxon if you're reading this in the future, but I'm leaving it in for now
            idx = df.loc[df['Labels'] == label, k+'_Index_Location'].iloc[0]
            if idx != 'No Match Found':
                unique_boxes = [int(x) for x in set(df['Bigram_idx'][label][idx])]
                if len(unique_boxes)!=1:
                    for i in unique_boxes:
                        bbox = df.loc[df['Labels'] == label, 'Bounding_Boxes'].iloc[0][i]
                        rect = cv2.boundingRect(bbox)
                        marking_color = get_color(df.loc[df['Labels'] == label, k+'_Similarity'].iloc[0])
                        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), marking_color, THICKNESS)
                        if found1 == False:
                            cv2.putText(img, "{} {} {}".format(k, df.loc[df['Labels'] == label, k+'_Prediction'].iloc[0],df.loc[df['Labels'] == label, k+'_Similarity'].iloc[0]), (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            found1 = True
                else:
                    bbox = df.loc[df['Labels'] == label, 'Bounding_Boxes'].iloc[0][unique_boxes[0]]
            #                 print(bbox)
                    rect = cv2.boundingRect(bbox)  # returns (x,y,w,h) of the rect
                    marking_color = get_color(df.loc[df['Labels'] == label, k+'_Similarity'].iloc[0])
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), marking_color, THICKNESS)
                    cv2.putText(img, "{} {} {}".format(k, df.loc[df['Labels'] == label, k+'_Prediction'].iloc[0],df.loc[df['Labels'] == label, k+'_Similarity'].iloc[0]), (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # display the image
    plt.imshow(img[:,:,::-1])
    plt.show()

def bounding_confidence_text(df,Taxon_truth,word_log_dic,label):

    """
    Given a dataframe, a label, and a dictionary of ground truth, this function displays the image associated with the label, along with bounding boxes for all the text in the image.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing information about the images and bounding boxes.
    label: The label for the image to be displayed.
    Taxon_truth (dict): A dictionary containing the ground truth for the taxon.
    word_log_dic (dict): A dictionary containing the label to image name mapping.
    
    Returns:
    None: This function does not return anything. It simply displays the image with bounding boxes and associated transcription and confidence.
    """
    print('GT: '+Taxon_truth[word_log_dic[label]])
    # get the image path for the specified label
    image_path = df.loc[df['Labels'] == label, 'Image_Path'].iloc[0]
    img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(80, 80))
    ax.imshow(img)
    for bbox,text,confidence in zip(df.loc[df['Labels'] == label, 'Bounding_Boxes'].iloc[0],
                                    df.loc[df['Labels'] == label, 'Transcription'].iloc[0],
                                    df.loc[df['Labels'] == label, 'Transcription_Confidence'].iloc[0]):
        rect = cv2.boundingRect(bbox)
        marking_color = get_color(confidence)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), marking_color, 2)
        cv2.putText(img, text+"  "+str(confidence), (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

   # display the image
    plt.imshow(img[:,:,::-1])
    plt.show()
    