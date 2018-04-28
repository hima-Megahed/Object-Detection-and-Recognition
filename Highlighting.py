from matplotlib import patches
from matplotlib.offsetbox import TextArea, AnnotationBbox
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


class Highlight:
    def __init__(self):
        self.RESULT_PATH = "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Results"
        self.Fig, self.AX = plt.subplots()
        self.ColorList = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#EC00FF", "#FFFB00", "#DAF7A6", "#FFC300"]
        os.chdir(self.RESULT_PATH)

    def HighlightObjects(self, img, res):
        self.AX.imshow(img['RealImage'])
        for i in range(len(img['ObjectImgsGrey'])):
            x = img['Xpos'][i]
            y = img['Ypos'][i]
            w = img['Width'][i]
            h = img['Height'][i]
            # Add the patch to the Axes
            self.AX.add_patch(
                patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=self.ColorList[i % 8],
                    fill=False      # remove background
                )
            )

            # Annotate the 1st position with a text box ('Test 1')
            offsetbox = TextArea(res[i], minimumdescent=False)

            ab = AnnotationBbox(offsetbox, (x, y),
                                xybox=(-20, 40),
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->"))
            self.AX.add_artist(ab)

        self.Fig.savefig(img['ImageName'], dpi=90, bbox_inches='tight')
        # plt.show()

