import numpy as np
import cv2

cam=cv2.VideoCapture(0)
facec=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data =[]
ix=0

while True:
    ret,fr=cam.read()
    if ret == True:
        gray= cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)  #bgr bcz it bydefault loads image in bgr format instead of rgb
        faces=facec.detectMultiScale(gray,1.3,5)



        
        for (x,y,w,h) in faces:
            fc=fr[y:y+h,x:x+w,:]  #bcz here rows are on the y axis that's why we use y first then x for image processing
            r = cv2.resize(fc,(50,50))  #here x is the no of colums then rows

            if ix%10==0 and len(data)<20:
                data.append(r)
            cv2.rectangle(fr,(x,y),(x+w,y+w),(0,0,255),2)  #image values are of int always bcz of pixel move from 0 to 255
        



        ix+=1    
        cv2.imshow('frame',fr)
        if cv2.waitKey(1)==27 or len(data)>=20:
            break
    else:
        print ("error")
        break
    
cv2.destroyAllWindows()
data=np.asarray(data)
print (data.shape)
np.save('face06',data)
