# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from skimage.feature import hog
from skimage.util.shape import view_as_windows
from numpy.linalg import norm
from collections import Counter

def extract_features(image_set, stride, hog_cell_size, patch_size, hog_block_size, orients):
    
    feature_set = []
    for image in image_set:

        window = (patch_size, patch_size)
        stride = stride
        image_cuts = view_as_windows(image, window, stride)
        patches = image_cuts.reshape((image_cuts.shape[0]**2, image_cuts.shape[2],image_cuts.shape[3] ))

        ## Obtain HOG for each patch
        hogs = []
        for patch in patches:
            hogs.append(hog(patch, orientations=orients, pixels_per_cell=(hog_cell_size, hog_cell_size),cells_per_block=(hog_block_size, hog_block_size), visualize=False, multichannel=False, feature_vector=True))
        hogs = np.vstack(hogs)
        
        feature_set.append((patches, hogs))

    return feature_set

def k_means_plus_plus(feature_vector, k):

    np.random.seed(7)

    # First centroid is just the first feature vector.
    centroids = [feature_vector[0]]

    for _ in range(1, k):
        # Calcuate probability based on the distance between points
        distance = np.array([min([np.inner(k-fv,k-fv) for k in centroids]) for fv in feature_vector])
        probs = distance/distance.sum()
        probs = probs.cumsum()
        
        # Rejection sampling to probabilistically pick centroid far away from existing
        r = np.random.rand()
        
        for j, p in enumerate(probs):
            if r < p:
                i = j
                break

        # Set the next centroid as the chosen vector from our feature vector space
        centroids.append(feature_vector[i])

    return np.array(centroids)

# CreateVisualDictionary() computes and save the visual dictionary
# The K-Means Clustering Method is used

def CreateVisualDictionary(feature_vector, k):

    print('Beginning clustering for k:' + str(k))

    vec_num = feature_vector.shape[0]
    vec_len = feature_vector.shape[1]

    converged = False

    ## STEP 1: Initialise centroids 
    # creates k centroid vectors of same length as our feature vectors
    centroids = k_means_plus_plus(feature_vector, k)

    #STEP 2: Iterate through update process until converged:
    j = 0   
    while(not converged):

        # Make a dictionary with centroid vector pairings. Gets reset every iteration
        clusters = {k: [] for k in range(centroids.shape[0])}

        j += 1
        # Step one - find the closest centroid for each point and group it. Do euclidian
        for i in range(vec_num):
            
            # Subtract the feature vector from each centroid, and calculate the l2 norm.
            # specify the axis of the length of the centroid vectors to sum over each
            norm_l2 = norm(centroids - feature_vector[i], axis=1)        
            # Take the minimum of this to find the closest centroid
            k = np.where(norm_l2==min(norm_l2))[0][0]

            clusters[k].append(i)

        # Store old value for convergence check
        old_centroids = centroids.copy()

        # Step two - update the centroid as the mean of the cluster
        for k in clusters:

            # Only update centroid vector if its corresponding cluster is non-zero in size
            if (len(clusters[k]) != 0):
                # Update  kth centroid with the mean of it's cluster 
                centroids[k] = np.mean([feature_vector[vec_index] for vec_index in clusters[k]], axis=0)
            else:
                # In case where no points associated with centroid, choose to re-assign it
                centroids[k] = feature_vector[random.randint(0,vec_len)]

        # Consider converged when no updates have been performed or the change in vectors is small
        old_centroids[old_centroids==0]=0.0000001 # fix div by 0
        converged =  (not np.any( (np.absolute(old_centroids - centroids)/old_centroids) > .01)) or (j>100)    
        print("Iteration " +  str(j) + " Complete")
        
        
    if j>100:
        print('converged due to iterations >100')
    else:
        print('converged due to threshold')

    return centroids

# Saves the most closest visual word to the mean of the cluster
def FindRepresentativePatch(centroids, feature_vector, k, patches):
     closest_visual_words = []
     for k in centroids:
          norm_l2 = norm(k - feature_vector, axis=1)   
          hog_index = np.where(norm_l2==min(norm_l2))[0][0]
          closest_visual_words.append(patches[hog_index])

     return (centroids, closest_visual_words)

# To create histogram for each of train and test images. 
# Using soft assignment is based on l2 norm (euclidean distance)

def ComputeHistogram(feature_set, visual_dictionary):
    h = [0 for k in visual_dictionary]
    for feature_vector in feature_set:

      norm_l2 = norm(feature_vector - visual_dictionary, axis=1)
      norm_l2[norm_l2==0] = 0.0001  # fix div by zero problem
      # Penalise by distance
      temp = ((1/norm_l2))**2

      h+= temp
    return h

# Compares two histograms and return distance
# Based on chi - squared 
def MatchHistogram(h1, h2):
    return np.sum( [((a-b)**2 / (a+b)) for (a,b) in zip(h1, h2)] )

# Loading dataset

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Data Preprocessing

test_data = test_data.reshape(10000, 28, 28).astype('float32')/255
train_data = train_data.reshape(60000, 28, 28).astype('float32')/255

# Extracting features from train and test sets
train_data_features = extract_features(train_data, stride=1, hog_cell_size=9, patch_size = 28, hog_block_size=1, orients=12)
test_data_features = extract_features(test_data, stride = 1, hog_cell_size = 9, patch_size = 28, hog_block_size = 1, orients = 12)

patches, features = zip(*train_data_features)
patches, features = (np.concatenate(patches), np.concatenate(features))

# Creating Visual Dictionary
centroids = CreateVisualDictionary(features, 100)

visual_dictionary_features, visual_dictionary_patches = FindRepresentativePatch(centroids, features, 100, patches)

for i in range(len(visual_dictionary_patches)):
   plt.imsave("./centroid_images/"+ str(i)+".jpeg", visual_dictionary_patches[i])

# computing histograms for train images

train_data_histograms = []
for _,feature_set in train_data_features:
  train_data_histograms.append(ComputeHistogram(feature_set, visual_dictionary_features))

# computing histogram for test images
test_data_histograms = []
for _,feature_set in test_data_features:
  test_data_histograms.append(ComputeHistogram(feature_set, visual_dictionary_features))

predicted_labels = []
for i in range(len(test_data_histograms)):
    histogram_distances = []

    # Find the closest training image:
    for j in range(len(train_data_histograms)):
        histogram_distances.append( (j, MatchHistogram(test_data_histograms[i], train_data_histograms[j])) )

    sorted_n = sorted(histogram_distances, key=lambda x: x[1])[:6]
    nn_labels = [train_labels[i[0]] for i in sorted_n]
    count = Counter(nn_labels)
    maxval = max(count.values())
    majority = [k for k,v in count.items() if v==maxval][0]
    predicted_labels.append(majority)

    print("label for image " + str(i) + " generated")

np.array(predicted_labels).tofile("./predicted_labels.csv", sep=',')

#  Calculating overall classification accuracy, class wise accuracy, precision and recall
predicted_labels = np.array(predicted_labels)
test_labels = np.array(test_labels)
N = test_labels.shape[0]

classes = [i for i in range(10)]

accuracy = 100*np.sum(predicted_labels == test_labels)/N

class_wise_accuracy, class_wise_precision, class_wise_recall = ([], [], [])
# cw_tp, cw_fp, cw_tn, cw_fn = ([], [], [], [])
x, y, z, w = ([], [], [], [])

for c in classes:
    
    tp = np.sum(predicted_labels[test_labels==c] == c)
    fp = np.sum(test_labels[predicted_labels==c] != c)
    tn = np.sum(predicted_labels[test_labels!=c] != c)
    fn = np.sum(predicted_labels[test_labels==c] != c)
    x.append(tp)
    w.append(fn)
    y.append(fp)
    z.append(tn)

    class_wise_accuracy.append(100*(tp/(tp+fn)))
    class_wise_precision.append(tp/(tp+fp))
    class_wise_recall.append(tp/(tp+fn))
  
precision = sum(class_wise_precision)/10
recall = sum(class_wise_recall)/10

stats = {'accuracy': accuracy, "class_wise_accuracy":class_wise_accuracy, "precision":precision, "recall":recall}

# Overall Precision and recall from average of classwise
print("\nOverall Performance")
print("Accuracy: " + str(round(stats['accuracy'], 2)) +  "  Precision: " + str(round(stats['precision'], 4)) + "  Recall: " + str(round(stats['recall'], 4)) + "\n\n")

print("Class wise accuracy")
classnames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(10):
    print("Class " + str(i) +  ": "  + str(round(stats['class_wise_accuracy'][i], 2)) + "% " + classnames[i])