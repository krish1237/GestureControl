import cv2

class FaceDetector():

    def __init__(self,faceCascadePath):
        self.faceCascade=cv2.CascadeClassifier(faceCascadePath)


    def detect(self, image, scaleFactor=1.1,
               minNeighbors=5,
               minSize=(30,30)):
        
        #function return rectangle coordinates of faces for given image
        rects=self.faceCascade.detectMultiScale(image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects

frontal_cascade_path="/home/krish/MPMC/project/haarCascadeTrained/haarcascade_frontalface_default.xml"
fd = FaceDetector(frontal_cascade_path)

class Cheese:

    def __init__(self,cam):
        self.cap = cv2.VideoCapture(0)
    
    def click(self):
        ret,frame = self.cap.read()
        #cv2.imshow('self.frame',self.frame) #not working in gtk 3+
        cv2.imwrite('./olio.png',frame)
        return frame

cam = Cheese(0)
img = cam.click()

def isFace(image):
    faces = list()
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = fd.detect(img_gray)
    if len(faces)!=0:
        return True
    else:
        return False

print(isFace(img))
