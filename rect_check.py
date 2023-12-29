import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

path_l = '/home/uas-laptop/Kantor_Lab/nerfstudio/temp/'
path_r = '/home/uas-laptop/Kantor_Lab/nerfstudio/temp_left/'

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,10)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

for i in range(5):
    imgL = cv2.imread(os.path.join(path_l, str(0000)+ str(i)+'.jpg'))
    imgR = cv2.imread(os.path.join(path_r, str(0000)+ str(i)+'.jpg'))
    imgL = cv2.imread(path_l + '0000' + str(i)+'.jpg')
    imgR = cv2.imread(path_r + '0000'+ str(i)+'.jpg')
    print(imgL.shape)
    print(imgR.shape)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL,None)
    kp2, des2 = sift.detectAndCompute(imgR,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    # now calculating epipolar lines
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print(F)
    
    img1 = imgL
    img2 = imgR
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    random_indices = np.random.choice(pts1.shape[0], 10)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1[random_indices],pts1[random_indices],pts2[random_indices])
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    #defining drawlines function
    
    img3,img4 = drawlines(img2,img1,lines2[random_indices],pts2[random_indices],pts1[random_indices])
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()