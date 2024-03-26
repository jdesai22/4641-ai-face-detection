import numpy as np
from sklearn import svm, metrics, datasets
from skimage import io, feature, filters, exposure, color
from skimage.io import imread_collection
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
import re, math
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def main():
    model = Image_Classification_Model()

    (raw_images, raw_labels) = model.load_data(r'face_images/') # insert directory here 
    train_raw, test_raw, train_labels, test_labels = train_test_split(raw_images, raw_labels, test_size=0.20, random_state=42)

    train_data = model.feature_extraction(train_raw)
    test_data = model.feature_extraction(test_raw)

    pca = PCA(n_components=0.85, svd_solver='full')
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)


    model.train(train_data, train_labels, 'rbf') # use an RBF kernel for this model
    predicted_labels = model.predict_labels(train_data)
    print("Training Results:")
    print("_________________________")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    predicted_labels = model.predict_labels(test_data)
    print("Test Results:")
    print("_________________________")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))



# TODO: add separate pre-processing method?
class Image_Classification_Model:
    def __init__(self):
        # stores the current classifier being used
        self.classifier = None
    
    # # helper method to read images 
    def imread_convert(self, f):
        # taken from https://scikit-image.org/docs/stable/api/skimage.io.html 
        return io.imread(f).astype(np.uint8)
    
    # loads the images from the given directory and returns arrays of data and labels
    def load_data(self, dir):
        data = []
        labels = []
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        data = io.ImageCollection(dir + '*.jpg', load_func=self.imread_convert)
        data = io.concatenate_images(data)
        for image in files:
            label = image.split('_')[0]
            if label == "easy" or label == "mid" or label == "hard":
                label = "fake"
            labels.append(label)
        return (data, labels)

    
    # perform feature extraction here
    def feature_extraction(self, data):
        feature_data = []
        for pic in data:
            feature_data.append(feature.hog(color.rgb2gray(pic), orientations=8, pixels_per_cell=(32, 32),
                                            cells_per_block=(3, 3)))
        return feature_data


    # train the model here: 
    # input_kernel refers to the specific kernel used in the SVM
    def train(self, train_data, train_labels, input_kernel):
        self.classifier = svm.SVC(kernel=input_kernel)
        self.classifier.fit(train_data, train_labels)
        pass

    # predict labels
    def predict_labels(self, data):
        return self.classifier.predict(data)



if __name__ == "__main__":
    main()