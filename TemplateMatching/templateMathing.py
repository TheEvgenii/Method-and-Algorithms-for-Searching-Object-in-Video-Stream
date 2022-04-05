#
#  TemplateMathing.py
#  Evgenii_Litvinov
#  COSC4399
#  Code was written in Python language
#  Created by Evgenii Litvinov on 9/28/21.
#  Reference: https://www.programmerall.com/article/5238458369/
#


import cv2 as cv
import numpy as np


def TemplateMathing():
    tpl = cv.imread('mouse1.png')
    target = cv.imread('worktable.jpg')

    #Definition 3 standard matching method
    methods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]

    #Take the height and width of the template picture
    tpl_h , tpl_w = tpl.shape[:2]
    nameofmethods = ['TM_SQDIFF_NORMED', 'TM_CCORR_NORMED','TM_CCOEFF_NORMED']

    i = 0

    for method in methods:
        result = cv.matchTemplate(target,tpl,method)
        #The returned minVal, maxVal (the minimum and maximum values ​​of pixel matching between the template and the target image), minLoc, maxLoc (the minimum and maximum positions)
        #Print out minVal, maxVal, minLoc, maxLoc: 0.0004309755750000477 1.0 (466, 185) (395, 327)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
        print(minVal, maxVal, minLoc, maxLoc)
        if method == cv.TM_SQDIFF_NORMED:
            tl = minLoc
        else:
            tl = maxLoc
        #  When using the first cv.TM_SQDIFF_NORMED matching method: br is the coordinates of the upper left corner of the rectangle
        #  When using the second cv.TM_CCORR_NORMED and the third cv.TM_CCOEFF_NORMED matching method: br is the coordinates of the upper left corner of the rectangle
        print(tl)
        br = (tl[0]+tpl_w,tl[1]+tpl_h)
        #Draw tl+br rectangle on the target picture, red, line width 2
        cv.rectangle(target,tl,br,(0,0,255),2)
        cv.namedWindow('match-'+np.str(method),cv.WINDOW_NORMAL)


        #h1, w1, = target.shape[:2]
        #h2, w2 = result.shape[:2]
        #img_3 = np.zeros((max(h1, h2), w1+w2,3), dtype=np.uint8)
        #img_3[:,:] = (255,255,255)
        #img_3[:h1, :w1,:3] = target
        #img_3[:h2, w1:w1+w2,:3] = result
        
        
        cv.imshow('match-'+np.str(i) + nameofmethods[i],target)
        cv.imshow('rusult-'+np.str(i) + nameofmethods[i],result)
        i +=1


TemplateMathing()

cv.waitKey(0)
cv.destroyAllWindows()