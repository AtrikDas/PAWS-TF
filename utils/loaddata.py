# Import all the required libraries
import numpy as np


def loadData(path):
    # Displaying the contents of the text file
    content = np.loadtxt(path)
    content = content.astype(int)

    def slice_per(source, step):
        return [source[i::step] for i in range(step)]

    array = slice_per(content, 3073)
    classes = array[0].astype(np.uint8)

    data = np.split(content, 501)
    data[0]

    pictures = [i[1:] for i in data]

    pictures = np.array(pictures)
    features = pictures.reshape(
        (len(pictures), 3, 32, 32)).transpose(0, 2, 3, 1)

    shuffler = np.random.permutation(len(features))
    features_shuffled = features[shuffler]
    classes_shuffled = classes[shuffler]

    X_train, X_test = np.split(
        features_shuffled, [int(len(features_shuffled)*0.8)])
    Y_train, Y_test = np.split(
        classes_shuffled, [int(len(classes_shuffled)*0.8)])

    return X_train, Y_train, X_test, Y_test
