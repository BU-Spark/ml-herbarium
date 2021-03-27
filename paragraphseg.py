import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def union(box):
    xcoords = [pt[0] for pt in box]
    ycoords = [pt[1] for pt in box]
    
    x = min(xcoords)
    y = min(ycoords)
    
    w = max(xcoords) - x
    h = max(ycoords) - y
    
    return (x, y, w, h)

def has_overlap2(b1,b2):
    l1 = [float('inf'), float('inf')]
    r1 = [float('-inf'), float('-inf')]
    l2 = [float('inf'), float('inf')]
    r2 = [float('-inf'), float('-inf')]

    for pt in b1:
        if pt[0]<l1[0] and pt[1]<l1[1]:
            l1[0]=pt[0]
            l1[1]=pt[1]
        elif pt[0]>r1[0] and pt[1]>r1[1]:
            r1[0]=pt[0]
            r1[1]=pt[1]
    
    for pt in b2:
        if pt[0]<l2[0] and pt[1]<l2[1]:
            l2[0]=pt[0]
            l2[1]=pt[1]
        elif pt[0]>r1[0] and pt[1]>r2[1]:
            r2[0]=pt[0]
            r2[1]=pt[1]
            
#     # e.g. in quadrants 2 and 4
#     if (r1[0]<l2[0] and r1[1]<l2[1]) or (r2[0]<l1[0] and r2[1]<l1[1]):
#         return False
    print(l1, r1)
    print(l2, r2)


    if r1[0]<l2[0] or r2[0]<l1[0] or r1[1]<l2[1] or r2[1]<l1[1]:
        return None

#     else:
    newL = [min(l1[0],l2[0]), min(l1[1],l2[1])]
    newR = [max(r1[0],r2[0]), max(r1[1],r2[1])]
    return newL, newR

#         return min(l1[0],l2[0]), min(l1[1],l2[1]), max(r1[0],r2[0])-min(l1[0],l2[0]), max(r1[1],r2[1])-min(l1[1],l2[1]) # x, y, w, h
    
def has_overlap(b1,b2):
    # boxes inputed as [[top left], [top right], [bottom right], [bottom left]]
    tl1, tr1, br1, bl1 = b1
    tl2, tr2, br2, bl2 = b2
    overlap = False
#     print(tl1, tr1, br1, bl1)
#     print(tl2, tr2, br2, bl2)


    if tl1[0]>=bl2[0] and tl1[0]<=br2[0] and tl1[1]>=tr2[1] and tl1[1]<=br2[1]: # if top left corner of 1 in 2
        overlap = True
    elif tr1[0]>=bl2[0] and tr1[0]<=br2[0] and tr1[1]>=tr2[1] and tr1[1]<=br2[1]: # if top right corner of 1 in 2
        overlap = True
    elif bl1[0]>=tl2[0] and bl1[0]<=tr2[0] and bl1[1]>=tr2[1] and bl1[1]<=br2[1]: # if bottom left corner of 1 in 2
        overlap = True
    elif br1[0]>=tl2[0] and br1[0]<=tr2[0] and br1[1]>=tr2[1] and br1[1]<=br2[1]: # if bottom right corner of 1 in 2
        overlap = True

    if overlap:
        newTL = [min(tl1[0],tl2[0]),min(tl1[1],tl2[1])]
        newTR = [max(tr1[0],tr2[0]),min(tr1[1],tr2[1])]
        newBL = [min(bl1[0],bl2[0]),max(bl1[1],bl2[1])]
        newBR = [max(br1[0],br2[0]),max(br1[1],br2[1])]
        return [newTL,newTR,newBR,newBL]


def expandBoxes(img_txt, xmarg=20,ymarg=40):
    
    boxes = []

    for i,box in enumerate(img_txt):
        coord = []
        temppt = []

        xcoords = []
        ycoords = []

        for j,val in enumerate(img_txt[i].split(",")):
            val = int(val)
            if j%2==0:
                temppt = []
                xcoords.append(val)

            temppt.append(val)

            if j%2!=0:
                coord.append(temppt)
                ycoords.append(val)

        # with the original box, expand it
        xavg = np.mean(xcoords)
        yavg = np.mean(ycoords)
        for k,(x,y) in enumerate(coord):
            if x>xavg:
                coord[k][0]+=xmarg
            else:
                coord[k][0]-=xmarg

            if y>yavg:
                coord[k][1]+=ymarg
            else:
                coord[k][1]-=ymarg

        boxes.append(coord)
        
    return boxes
