# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:05:49 2018

@author: Andre
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math



def gaussian(x,sigma):
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(x**2)/(2*(sigma**2)))

def gauss(x,sigma):
    return math.exp((-0.5)*((x/sigma)**2))

def mod(x,y):
    return math.sqrt(x**2 + y**2)

def bilateralFilter(src,d,sigmaColor):
    #sigma values to change size of effect
    #<10 is little, >150 is very large (almost cartoonish)
    #d is filter size, large if >2 and slower, reccomended 5 for realtime, 4 for offline
    img = cv.imread(src)
    out = cv.imread(src)

    gaussDist = [[0]]
    sdArray = []
    for i in range(0,d+1):
        for j in range(1,d+1):
            pair = []
            dist = mod(i,j)
            for k in range(0,4):
                sdArray.append(dist)
            pair.append(dist)
            if pair not in gaussDist:
                gaussDist.append(pair)
    sigmaSpace = np.std(sdArray)
    for g in gaussDist:
        g.append(gaussian(g[0],sigmaSpace))

    gaussCol = []
    for i in range(0,256):
        gaussCol.append(gaussian(i,sigmaColor))
        
    
    for i in range(0,len(img)):
        for j in range(0,len(img[0])):
            norm = [0,0,0]
            iFilter = [0,0,0]
            minX = i - d
            minY = j - d
            maxX = i + d
            maxY = j + d
            if minX < 0:
                minX = 0
            if minY < 0:
                minY = 0
            if maxX >= len(img):
                maxX = len(img)-1
            if maxY >= len(img[0]):
                maxY = len(img[0])-1

            neighborhood = []
            neighborhoodColor = []
            for k in range(minX,maxX+1):
                for l in range(minY,maxY+1):
                    neighborhood.append([k,l])
                    neighborhoodColor.append(img[k][l])
                    

            for n in neighborhood:
                x,y = n[0],n[1]
                space = mod(x-i,y-j)
                for g in gaussDist:
                    if g[0] == space:
                        gSpace = g[1]
            
                        
                color = [0,0,0]
                for c in range(0,3):
                    color[c] = img[x][y][c] - img[i][j][c]
                    if color[c] < 0:
                        color[c] = -color[c]
                    prod = gSpace * gaussCol[color[c]]
                    norm[c] += prod
                    iFilter[c] += prod * img[x][y][c]
            print(norm)
            

            for k in range(0,3):
                iFilter[k] = iFilter[k]/norm[k]

            out[i][j] = iFilter
            print(img[i][j], out[i][j], i, j)

            spaceArray = np.sqrt(np.sum(np.subtract(np.array(neighborhood),[i,j])**2,axis=1))
            #print(spaceArray)
            a = np.where(spaceArray == np.take(np.array(gaussDist), [0], axis = 1))
            #print(a)
            gSpaceArray = np.take(np.array(gaussDist),a[0],axis = 0)
            print(gSpaceArray)
            print(np.take(gSpaceArray,[1],axis=0))
            colorDelta = np.diff(([np.array(neighborhoodColor),img[i][j]]),axis=0)
            #print(colorDelta)
            prod = np.multiply(np.take(gSpaceArray,[1],axis=0),colorDelta)
            norm = np.sum(prod, axis=1)
            #print(norm)
                

    #cv.imshow('image',out)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return out
#bilateralFilter('test3a.jpg',3,1.5)

def jointBilateralFilter(noFlash,flash,d,sigmaColor):
    #sigma values to change size of effect
    #<10 is little, >150 is very large (almost cartoonish)
    #d is filter size, large if >2 and slower, reccomended 5 for realtime, 4 for offline
    nf = cv.imread(noFlash)
    f = cv.imread(flash)
    out = cv.imread(flash)

    gaussDist = [[0]]
    sdArray = []
    for i in range(0,d+1):
        for j in range(1,d+1):
            pair = []
            dist = mod(i,j)
            for k in range(0,4):
                sdArray.append(dist)
            pair.append(dist)
            if pair not in gaussDist:
                gaussDist.append(pair)
    sigmaSpace = np.std(sdArray)
    for g in gaussDist:
        g.append(gaussian(g[0],sigmaSpace))

    gaussCol = []
    for i in range(0,256):
        gaussCol.append(gaussian(i,sigmaColor))

    
    for i in range(0,len(nf)):
        for j in range(0,len(nf[0])):
            norm = [0,0,0]
            iFilter = [0,0,0]
            minX = i - d
            minY = j - d
            maxX = i + d
            maxY = j + d
            if minX < 0:
                minX = 0
            if minY < 0:
                minY = 0
            if maxX >= len(nf):
                maxX = len(nf)-1
            if maxY >= len(nf[0]):
                maxY = len(nf[0])-1

            neighborhood = []
            for k in range(minX,maxX+1):
                for l in range(minY,maxY+1):
                    neighborhood.append([k,l])

            for n in neighborhood:
                x,y = n[0],n[1]
                space = mod(x-i,y-j)
                for g in gaussDist:
                    if g[0] == space:
                        gSpace = g[1]
            
                        
                color = [0,0,0]
                for c in range(0,3):
                    color[c] = f[x][y][c] - f[i][j][c]
                    if color[c] < 0:
                        color[c] = -color[c]
                    prod = gSpace * gaussCol[color[c]]
                    norm[c] += prod
                    iFilter[c] += prod * nf[x][y][c]

            for k in range(0,3):
                iFilter[k] = iFilter[k]/norm[k]

            out[i][j] = iFilter
            print(nf[i][j], out[i][j], i, j)
                

    #cv.imshow('image',out)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return out

#crossBilateralFilter('test3a.jpg','test3b.jpg',3,30,30)
#bilateralFilter('test3a.jpg',3,1.5,1.5)

#img = cv.bilateralFilter(cv.imread('test3a.jpg'),5,150,150)
#cv.waitKey(0)
#cv.destroyAllWindows()
#cv.imwrite('test3a-5-10-10-cv2.jpg',img)

img = jointBilateralFilter('test3a.jpg','test3b.jpg',12,12)
cv.imwrite('joint-25-12.jpg',img)



#img = cv.bilateralFilter(cv.imread('test2.png'),5,10,10)
#cv.imwrite('test2-5-10-10.png',img)
#img = cv.bilateralFilter(cv.imread('test2.png'),5,1,150)
#cv.imwrite('test2-5-1-150.png',img)
#img = cv.bilateralFilter(cv.imread('test2.png'),5,150,1)
#cv.imwrite('test2-5-150-1.png',img)
#img = cv.bilateralFilter(cv.imread('test2.png'),5,150,150)
#cv.imwrite('test2-5-150-150.png',img)
#img = cv.bilateralFilter(cv.imread('test2.png'),5,50,50)
#cv.imwrite('test2-0-150-150.png',cv.bilateralFilter(cv.imread('test2.png'),0,150,150))
