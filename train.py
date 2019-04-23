import numpy as np
import sklearn
import scipy
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage.morphology import binary_closing, binary_dilation
from skimage import io, exposure
import matplotlib.pyplot as plt


def train(file, print_bool=False, test_bool=False):

    img = io.imread(file)

    # Binarize image
    th = 205
    img_binary = (img < th).astype(np.double)

    # Morpohology
    # img_binary = binary_dilation(img_binary)
    # img_binary = binary_closing(img_binary)

    # Connected component analysis
    img_label = label(img_binary, background=0)

    # Bounding boxes and moments
    regions = regionprops(img_label)
    if print_bool:
        io.imshow(img_binary)
    ax = plt.gca()
    if print_bool:
        plt.title('Bounding boxes')

    Features = []

    count = 0
    test_regions = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        # Note: "minr - 2" for better fitting box
        minr = minr - 2
        width = maxc - minc
        height = maxr - minr
        if 10 < width < 100 and 10 < height < 100:
            # Add rectangle
            if print_bool:
                ax.add_patch(plt.Rectangle((minc, minr), width, height, fill=False, edgecolor='red', linewidth=1))

            # Hu Moments
            roi = img_binary[minr:maxr, minc:maxc]
            m = moments(roi)
            cr = m[0, 1] / m[0, 0]
            cc = m[1, 0] / m[0, 0]
            mu = moments_central(roi, (cr, cc))
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            count += 1

            test_regions.append(props)

    if print_bool:
        io.show()

    if test_bool:
        return Features, test_regions
    return Features


