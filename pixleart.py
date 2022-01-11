import skimage 
from sklearn.cluster import KMeans
from numpy import linalg as LA
import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread(".\mario.jpg")


def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]
            
    return imgC

def segmentImgClrRGB(img, k):
    imgC = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    imgC.shape = (img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    kmeans.shape = (h, w)
    return kmeans

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)

def pixelate(img, w, h):
    height, width = img.shape[:2]
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

img16 =(pixelate(img, 64, 64)) 
img17= kMeansImage(img16,10)

cv2.imwrite(".\mario2.jpg",img17)

