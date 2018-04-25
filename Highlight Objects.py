from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

TESTING_PATH = "F:\GitHub - Projects" \
                               "\Object-Detection-and-Recognition\Data set" \
                               "\Custom Testing"
img = misc.imread(TESTING_PATH + "\Cat Test1.png")


"""
unique, counts = np.unique(img.reshape(-1, img.shape[2]), return_counts=True, axis=0)


inds = counts.argsort()
inds = list(reversed(inds))
unique = unique[inds]
counts = counts[inds]

# img = plt.imread(TRAINING_PATH + "\Model2 - Cat.jpg", as_grey=True)
fig, ax = plt.subplots()

ax.imshow(img)

"""
'''
# Create a Rectangle patch
#rect = patches.Rectangle((50,300),60,60,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
#ax.add_patch(rect)
ax.add_patch(
    patches.Rectangle(
        (50, 100),
        60,
        60,
        linewidth=2,
        edgecolor='r',
        fill=False      # remove background
    )
)

# Annotate the 1st position with a text box ('Test 1')
offsetbox = TextArea("Test 1", minimumdescent=False)

ab = AnnotationBbox(offsetbox, (50,100),
                    xybox=(-20, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

ax.add_patch(
    patches.Rectangle(
        (130, 100),
        60,
        60,
        linewidth=2,
        edgecolor='r',
        fill=False      # remove background
    )
)

ab2 = AnnotationBbox(offsetbox, (130,100),
                    xybox=(-20, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab2)

fig.savefig('F:\\GitHub - Projects\\Object-Detection-and-Recognition\\rect2.png', dpi=90, bbox_inches='tight')
'''
# plt.show()
