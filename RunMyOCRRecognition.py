import train
import normalize
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage import io
import matplotlib.pyplot as plt
import pickle


def training_phase():

    # the letter array is for the names of bmp files with their respective letters being trained
    # although much of this is hard-coded into the program, it can be trained to read any character!
    
    letters = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']

    Features = []
    Labels = []

    for letter in letters:
        file = letter + ".bmp"
        new = train.train(file)
        count = 0
        for feature in new:
            count += 1
            Features.append(feature)
        for a in range(0, count):
            Labels.append(letter)

    return Features, Labels


def get_closest_neighbor(D, Labels):

    # Finding closest neighbor
    correct = 0
    min_index = 0
    predictions = []

    for a in range(len(D)):
        min = D[0][0] + 1000
        for b in range(len(D)):
            if 0 < D[a][b] < min:
                min = D[a][b]
                min_index = b
        predictions.append(Labels[min_index])
        if Labels[min_index] == Labels[a]:
            correct += 1

    return correct, predictions


def print_stats(correct, total, labels, predictions):

    # amount correct fraction
    print "Fraction: ", correct, "/", total

    # amount correct percentage
    correct = correct * 100.0
    print "Percentage: ", correct / total


def show_confusion(labels, predictions):
    # Finding confusion matrix
    confM = confusion_matrix(labels, predictions)
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()
    print "[a, d, m, n, o, p, q, r, u, w]"
    for row in confM:
        print row


def training_recog(features, labels):

    D = cdist(features, features)

    temp = get_closest_neighbor(D, labels)

    correct = temp[0]
    predictions = temp[1]

    total = len(labels)
    print_stats(correct, total, labels, predictions)
    show_confusion(labels, predictions)


def test1_recog(file, features, labels, avg, std_dev):

    test_components = train.train(file, True, True)
    initial_test_features = test_components[0]
    test_regions = test_components[1]
    initial_test_features = normalize.get_standard_distribution(initial_test_features, avg, std_dev)
    test_length = len(test_regions)

    pkl_name = file[:file.index(".")] + "_gt.pkl.txt"

    pkl_file = open(pkl_name, 'rb')
    dictionary = pickle.load(pkl_file)
    pkl_file.close()
    classes = dictionary['classes']
    locations = dictionary['locations']

    # Clear unwanted boxes/features
    # For every region, we see if it contains a particular center coordinate.
    # If it does, its feature is valid.
    matches = []
    for region in test_regions:
        minr, minc, maxr, maxc = region.bbox
        matched = False
        for location in locations:
            row = location[1]
            col = location[0]
            if minr < row < maxr and minc < col < maxc:
                matched = True
                break
        matches.append(matched)

    test_features = []
    for num in range(test_length):
        if matches[num]:
            test_features.append(initial_test_features[num])

    dist = cdist(test_features, features)
    d_index = np.argsort(dist, axis=1)

    predictions = []
    for index in d_index:
        predictions.append(labels[index[0]])

    correct = 0.0
    total = len(classes)
    for num in range(total):
        if predictions[num] == classes[num]:
            correct += 1
    print correct/total


def run(test):

    # Training: obtain features with training data
    temp = training_phase()
    features = temp[0]
    labels = temp[1]

    avg = normalize.get_average(features)

    std_dev = normalize.get_standard_deviation(features, avg)

    features = normalize.get_standard_distribution(features, avg, std_dev)

    test1_recog(test, features, labels, avg, std_dev)

run("test1.bmp")
