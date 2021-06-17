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

    print(features)
    print(classes)
    return features, classes
