# Instructions


######## Update the following parameters for yourself ########
### Update the following parameters for yourself 

# inputDir : The path for batch image data it would be a good 
# idea to put absolute path here,then no need to worry about 
# where this script is saved. data type: string

# outputDir : the destination to save the output images. 
# data type: string

# totalSoybeansPerImg : this parameter is used to differentiate 
# images with few soybeans from images with too many soybeans 
# (like more than 100 soybeans per img ). data type: int

# colorPad : whether images contains colorPad. data type: boolean

# targetCmPerPixel : To normalize size of soybeans over all 
# source images. data type:float
###############################################################


## update the following parameters
inputDir = '/Users/huiminhan/Desktop/InfoVis/CropAI/RawData/Pod_photos_20210512_Batch2/'
outputDir = '/Users/huiminhan/Desktop/InfoVis/CropAI/CroppedImageData/Crop_batch2/'
totalSoybeansPerImg = 15 # could be an approxiate value
colorPad = True
targetCmPerPixel = 0.007

import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
from imutils import perspective
import math

def resize_image(image, height, width):  
    h, w, _ = image.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest_edge = max(h, w)
    if h < w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w, _ = image.shape
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass  
    BLACK = [0,0,0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    h,w, _ = constant.shape
    if h <= height and w <= width:
        resized = cv2.copyMakeBorder(constant, (height-h)//2,(height-h)//2, (width-w)//2, (width-w)//2, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        resized = constant
    return resized

def maxCont(contours):
    contAreaList = []
    for cont in contours:
        area = cv2.contourArea(cont)
        contAreaList.append(area)
    return max(contAreaList)

def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

def cmPerPixel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edge_detect = cv2.Canny(gray, 60, 200) 
    edge_detect = cv2.dilate(edge_detect, None, iterations=1)
    edge_detect = cv2.erode(edge_detect, None, iterations=1)
    cnts = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for cont in cnts:
        ares = cv2.contourArea(cont)  
        if ares == maxCont(cnts):
            rect = cv2.boundingRect(cont)
            bbox = cv2.minAreaRect(cont)
            bbox = cv2.boxPoints(bbox) 
            bbox = np.array(bbox, dtype="int")
            ## order the contours and draw bounding box
            bbox = perspective.order_points(bbox)
            for (x, y) in bbox:
                (tl, tr, br, bl) = bbox
                (tltrX, tltrY) = mdpt(tl, tr)
                (blbrX, blbrY) = mdpt(bl, br)
                (tlblX, tlblY) = mdpt(tl, bl)
                (trbrX, trbrY) = mdpt(tr, br)
            ## compute the Euclidean distances between the mdpts
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            cmPerPixel = 20/dA
    print("The cm per pixel: ", cmPerPixel)
    return cmPerPixel

def Nrotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return (nRotatex, nRotatey)

def Srotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return (sRotatex,sRotatey)

def rotatecordiate(angle,rectboxs,pointx,pointy):
    output = []
    for rectbox in rectboxs:
        if angle>0:
            output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
        else:
            output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
    return output

def imageCrop(image,box):
    xs = [x[1] for x in box]
    ys = [x[0] for x in box]
    cropimage = image[min(xs):max(xs),min(ys):max(ys)]
    return cropimage
  
def normalizeSize(image,cmPerPixel,targetCmPerPixel):
    scale_percent = cmPerPixel/targetCmPerPixel      # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image
    
def crop(image,outputDir,imgNum,target_height,target_width,targetPerPixel=0.007,imgId='',colorPad=False):
    src = cv2.imread(image)
    cmperPixel = cmPerPixel(src)
    height = target_height
    width = target_width
    fsrc = np.array(src,dtype = np.float32)/255.0
    (b,g,r) = cv2.split(fsrc)
    gray = 2 * g - b - r + 0.3 * (1.4 * r - b)
    
    ## get min and max
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    
    ## image processing to remove noise
    gray_u8 = np.array((gray - minVal)/ (maxVal - minVal) * 255, dtype = np.uint8)
    (_, thresh) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    if imgNum == 15:
        size = 100
    elif imgNum >15:
        size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    ## convert the gray image into colorful
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & closed, g8 & closed, r8 & closed]) 
    
    ## find contours
    contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count=0 
    WidthList = [cv2.boundingRect(cont)[2] for cont in contours]
    HeightList = [cv2.boundingRect(cont)[3] for cont in contours]
    longestEdge = max(max(WidthList),max(HeightList))+20
#     print(longestEdge)
    for cont in contours:       
        ## calculate the area size of contours
        ares = cv2.contourArea(cont)
     
        ## filter contours not for soybeans
        if ares<10000: 
            continue       
        elif ares==maxCont(contours) and colorPad==True:
            continue    
        count+=1 
        rect = cv2.minAreaRect(cont) #get the coordinates
        box_origin = cv2.boxPoints(rect)
#         print(rect[2])
        box = rotatecordiate(rect[2],box_origin,rect[0][0],rect[0][1])
        M = cv2.getRotationMatrix2D(rect[0],rect[2],1)
        dst = cv2.warpAffine(color_img,M,(2*color_img.shape[0],2*color_img.shape[1]))
        new_img = imageCrop(dst,np.int0(box))
        normalizeSize_img = (normalizeSize(new_img,cmperPixel,targetCmPerPixel))
        resized_img = resize_image(normalizeSize_img,height,width)       
#         ## get the coordinates to crop
#         x,y,w,h  = cv2.boundingRect(cont)

        ## crop the image and save new image
        if os.path.isdir(outputDir):
            pass
        else:
            os.mkdir(outputDir)
#         new_img=color_img[y-10:y+h+10,x-10:x+w+10]
#         resized_img = resize_image(new_img,height,width)

        ## writes the new file in the Crops folder
        cv2.imwrite(outputDir+'croped_'+str(imgId)+ '_' + str(count)+ '.jpg', resized_img)
        
def batchProcessing(inputDir,outputDir,imgNum,target_height=900,target_width=900,colorPad=False):
    imread_failed = []
    imgNum = imgNum
    height = target_height
    width = target_width
    pad=colorPad
    for (path,dirname,filenames) in os.walk(inputDir):
        for image in filenames:
            try:
                src = path + image
                imgId = image.split('.')[0]
                crop(src,outputDir,imgNum,height,width,imgId=imgId,colorPad=pad)  
                print("Image "+str(image)+" cropped.")
            except:
                imread_failed.append(image)
    
    print("Failed images: ",imread_failed)

def splitData(inputDir):
    soybeans_ready = []
    soybeans_late = []
    soybeans_disease = []
    for *_, filenames in os.walk(inputDir):
        for file in filenames:
            if file.split('_')[-2].endswith('ready'):
                soybeans_ready.append(file)
            elif file.split('_')[-2].endswith('late'):
                soybeans_late.append(file)
            elif file.split('_')[-2].endswith('disease'):
                soybeans_disease.append(file)
            else:
                print('File name incorrect')
    return soybeans_ready,soybeans_late,soybeans_disease   

if __name__ == "__main__":
    batchProcessing(inputDir,outputDir,totalSoybeansPerImg,target_height=1200,target_width=1200,colorPad=colorPad)
