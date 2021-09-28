# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:05:49 2018

@author: Andre
"""

import cv2 as cv
import numpy as np
import math

def gaussian(x,sigma):
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(x**2)/(2*(sigma**2)))

def mod(x,y):
    return math.sqrt(x**2 + y**2)

def jointBilateralFilter(noFlash,flash,d,sigmaColor):
    try:
        nf = cv.imread(noFlash)
        f = cv.imread(flash)
        out = cv.imread(flash)
    except:
        print('Please check your filenames')
        return

    gaussDist = [[0]]
    sdArray = []
    for i in range(-d,d+1):     # Generate an array of all possible distances and their corresponding gaussian
        for j in range(-d,d+1):
            pair = []
            dist = mod(i,j)
            sdArray.append(dist)
            pair.append(dist)
            if pair not in gaussDist:
                gaussDist.append(pair)
    sigmaSpace = np.std(sdArray)
    for g in gaussDist:
        g.append(gaussian(g[0],sigmaSpace))

    gaussCol = []
    for i in range(0,256):      # Generate an array of all possible intensity differences and their corresponding gaussian
        gaussCol.append(gaussian(i,sigmaColor))


    for i in range(0,len(nf)):      # Iterate through every pixel
        for j in range(0,len(nf[0])):
            norm = [0,0,0]
            iFilter = [0,0,0]

            minX = i - d            # Set window size
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

            neighborhood = []       # Create pixel neighborhood from window
            for k in range(minX,maxX+1):
                for l in range(minY,maxY+1):
                    neighborhood.append([k,l])

            for n in neighborhood:      # Calculate function of space and intensity for all pixels in the neighborhood and sum
                x,y = n[0],n[1]
                space = mod(x-i,y-j)
                for g in gaussDist:
                    if g[0] == space:
                        gSpace = g[1]


                color = [0,0,0]
                for c in range(0,3):
                    color[c] = int(f[x][y][c]) - int(f[i][j][c])
                    if color[c] < 0:
                        color[c] = -color[c]
                    prod = gSpace * gaussCol[color[c]]
                    norm[c] += prod
                    iFilter[c] += prod * nf[x][y][c]

            for k in range(0,3):
                iFilter[k] = iFilter[k]/norm[k]

            out[i][j] = iFilter     # Set output pixel (i,j) to be filtered pixel
            print(nf[i][j], out[i][j], i, j) #print after each pixel processed

    ext = noFlash.find('.')
    name = str(noFlash[0:ext] + '-' + str(d) + '-' + str(sigmaColor) + noFlash[ext:])

    #cv.imwrite(name, out)      # Automated output to file
    #cv.imshow(name,out)        # Show image after processed
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return out
