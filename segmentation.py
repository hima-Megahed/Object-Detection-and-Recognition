import cv2
import numpy as np
import glob
import os

class SegmentationEngine:

    def __init__(self):

        self.idx = 0
        self.ColoredImages = [cv2.imread(file) for file in glob.glob(
            "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Testing\*.png")]
        self.RealImages = [cv2.imread(file) for file in glob.glob(
            "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Testing\*.jpg")]
        self.RealImagesGrey = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(
            "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Testing\*.jpg")]
        self.ImagesNames = [file.replace("F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Testing\\", "") for file
                               in glob.glob(
                "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Testing\*.png")]
        os.chdir("F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Custom Testing")
        self.TestImages = list()


    def segmentationAlgo(self, coloredImage,RealImage, RealImageGrey):

        # cur = "C:/Users/Mohamed/Desktop/neural task/Data set/Testing/T6 - Helicopter.png"

        # image = cv2.imread(cur)
        image=coloredImage
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.filter2D(image, -3, kernel)
        edged = cv2.Canny(gray, 0, 255)
        kernel = np.ones((2, 2), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        WHITE = [255, 255, 255]
        edged = cv2.copyMakeBorder(edged, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        Xpos = []
        Ypos = []
        Width = []
        Height = []
        ImgObjects = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 20:
                self.idx+=1
                new_img=RealImage[y:y+h,x:x+w]
                new_img_grey = RealImageGrey[y:y+h,x:x+w]

                cv2.imwrite(str(self.idx) + ".png", new_img)
                Xpos.append(x)
                Ypos.append(y)
                Width.append(w)
                Height.append(h)
                ImgObjects.append(new_img_grey)
        return Xpos, Ypos, Width, Height, ImgObjects

    def get_tests(self):
        for i in range(len(self.ColoredImages)):
            Xpos, Ypos, Width, Height, ImgObjects = self.segmentationAlgo(
                self.ColoredImages[i], self.RealImages[i], self.RealImagesGrey[i])
            self.TestImages.append({
                'RealImage':self.RealImages[i],
                'RealImageGrey':self.RealImagesGrey[i],
                'ObjectImgsGrey':ImgObjects,
                'ImageName':self.ImagesNames[i],
                'Xpos':Xpos,
                'Ypos':Ypos,
                'Width':Width,
                'Height':Height
            })
        return self.TestImages