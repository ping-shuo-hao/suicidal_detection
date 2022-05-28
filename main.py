import cv2
import numpy as np

confindenceTh=0.45
nmsTh=0.3
bth=1.0
def findPerson(outputs,img):
    ht,wt,ct=img.shape
    bbox=[]
    confs=[]
    for output in outputs:
        for det in output:
            confidence=det[5]
            if confidence>confindenceTh:
                w,h=int(det[2]*wt),int(det[3]*ht)
                x,y=int(det[0]*wt-w/2),int(det[1]*ht-h/2)
                ar=w/h
                bbox.append([x, y, w, h])
                confs.append(float(confidence))
    indicies=cv2.dnn.NMSBoxes(bbox,confs,confindenceTh,nmsTh)
    for i in indicies:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        if ar<bth:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), 2)
            cv2.putText(img, f'Person {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 128, 0), 2)
           # cv2.putText(img,f'Person {int(confs[i]*100)}% {ar}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,128,0),2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f'Person with suicidal intent {int(confs[i] * 100)}%', (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #cv2.putText(img, f'Person with suicidal intent {int(confs[i] * 100)}% {ar}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)




cap=cv2.VideoCapture("test3.mp4")
wht=320
classfile='coco.names'
classname=[]
with open(classfile,'rt') as f:
    classname=f.read().rstrip('\n').split('\n')
Configuration='yolov3.cfg'
Weights='yolov3.weights'
net=cv2.dnn.readNetFromDarknet(Configuration,Weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
imgl=[]
len=0
while True:
    success,img=cap.read()
    if success==False:break
    blob=cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)
    layernames=net.getLayerNames()
    outputNames=[layernames[i[0]-1]for i in net.getUnconnectedOutLayers()]
    outputs=net.forward(outputNames)
    findPerson(outputs,img)
    imgl.append(img)
    len=len+1
#    cv2.imshow('image',img)
#    cv2.waitKey(1)
height,width,layers=imgl[1].shape
video=cv2.VideoWriter('video.mp4',-1,1,(width,height))
for i in range(0,len):
    video.write(imgl[i])
cap.release()
cv2.destroyAllWindows()
video.release()