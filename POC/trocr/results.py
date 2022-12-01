import cv2
import matplotlib.pyplot as plt
# Functions for Displaying Predictions


def get_color(similarity):
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
        if k == 'Taxon' or k == 'Countries':
            idx = df.loc[df['Labels'] == label, k+'_Index_Location'].iloc[0]
        #             print(idx)
            if idx != 'No Match Found':
        #                 print(df.loc[df['Labels'] == label, 'Bounding_Boxes'].iloc[0])
                bbox = df.loc[df['Labels'] == label, 'Bounding_Boxes'].iloc[0][idx]
        #                 print(bbox)
                rect = cv2.boundingRect(bbox)  # returns (x,y,w,h) of the rect
                marking_color = get_color(df.loc[df['Labels'] == label, k+'_Similarity'].iloc[0])
                cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), marking_color, THICKNESS)
                cv2.putText(img, k, (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # display the image
    plt.imshow(img[:,:,::-1])
    plt.show()

def bounding_confidence_text(df,Taxon_truth,word_log_dic,label):
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
    