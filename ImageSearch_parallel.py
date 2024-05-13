import os
import sys
import threading
import concurrent.futures
import time
import cv2
import matplotlib.pyplot as plt
from FeatureVectors import FeatureVectors
from QuerySearch import QuerySearch


def extractFeatureVectors(image_path):
    # Extracts feature vectors for input image

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 500))
    featureVectors = FeatureVectors(image)
    vectors = featureVectors.getFeatureVector()

    imageName = image_path.split("/")[-1]
    return [imageName, vectors]


def ThreadedFeatureExtraction(images_list):
     # Performing feature extraction of databse images using multithreading

    features = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(extractFeatureVectors, image_path)
                   for image_path in images_list]

    for f in concurrent.futures.as_completed(results):
        imageName, vectors = f.result()
        features[imageName] = vectors

    return features


def getImg(img):
    image_db_path = "dataset/jpg/"
    image = cv2.imread(image_db_path+img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def ImageSearch(queryImage, feature_map):
    # Performs Image Search using Query image

    image_db_path = "dataset/jpg/"
    image_paths = []
    for img in os.listdir(image_db_path):
        image_paths.append(image_db_path+img)

    # print(feature_map)
    if feature_map == {}:
        print('inside')
        features = ThreadedFeatureExtraction(image_paths)
        feature_map = features
    # print(feature_map)
    queryImage_path = "test/"+queryImage
    imageName, queryVector = extractFeatureVectors(queryImage_path)

    search = QuerySearch(queryVector, feature_map)

    results = search.performSearchParallel()
    results = sorted(results, key=lambda x: x[1])

    return results, feature_map
