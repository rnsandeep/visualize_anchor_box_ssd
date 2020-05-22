import os, shutil, sys

import os
from xml.dom import minidom
os.environ['GLOG_minloglevel'] = '2'
import  os, sys, cv2
import argparse
import shutil
import sys ,os

def files_with_ext(mypath, ext):
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and os.path.splitext(os.path.join(mypath, f))[1] == ext]
    return files




def read_xml(xml_path):
    flag = True
    doc = minidom.parse(xml_path)
    objects = doc.getElementsByTagName("object")
    fname = doc.getElementsByTagName("filename")[0].firstChild.data
    size = doc.getElementsByTagName("size")[0]
    width = size.getElementsByTagName("width")[0].firstChild.data
    height =  size.getElementsByTagName("height")[0].firstChild.data
    labelledObjects = dict()

    for obj in objects:
        name = obj.getElementsByTagName("name")[0].firstChild.data
        xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
        ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
        xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
        ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
       
        
        if int(ymin) >= int(height) or int(ymax) >= int(height) or int(xmin) >= int(xmax) or int(ymin) >= int(ymax) or int(xmin) >= int(width) or int(xmax) >= int(width) :
              flag = False
              continue
        if int(xmin) <=0 or int(ymin) <=0 or int(xmax) <=0 or int(ymax) <=0:
              flag = False
              continue

        if (int(xmax)- int(xmin))*(int(ymax)-int(ymin)) <= 0: 
              flag = False
              continue
   
        box = [xmin, ymin, xmax, ymax]
        bb = []
        for b in box:
            if int(b)  ==0:
                b = 1
            bb.append(b)
        height = int(height)
        width = int(width)
        box = (int(bb[0])*1.0/width, int(bb[1])*1.0/height, int(bb[2])*1.0/width , int(bb[3])*1.0/height)

        box = [int(i*300) for i in box]


        if name not in labelledObjects:
            labelledObjects[name] = []
            labelledObjects[name].append(box)
        else:
            labelledObjects[name].append(box)

    if len(labelledObjects.keys())== 0:
        return False, xml_path

    return labelledObjects, width, height


def drawImage(image, xml, output_path):
    I = cv2.imread(image)
    objs = readXML(xml)
    for obj in objs:
        boxes = objs[obj]
        for box in boxes:
            color = (255, 0, 0)  
            thickness = 2
            # Using cv2.rectangle() method 
            # Draw a rectangle with blue line borders of thickness of 2 px 
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            image = cv2.rectangle(I, start_point, end_point, color, thickness)
    cv2.imwrite(os.path.join(output_path, os.path.basename(image)), image)
    return 


