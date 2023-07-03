import numpy as np
import cv2
import matplotlib.pyplot as plt


def getcentroids(data,k,centroids):
     # calculate distances of each point to the centroids
    distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in centroids]) #eucledian distance

    # assign each point to the closest centroid
    clusters = np.argmin(distances, axis=0) 

    new_centroids=[0]*k    # calculate new centroids
   
    for i in range(k):
        new_centroids[i] = np.mean(data[clusters == i],axis=0,dtype='int64') #calculating the mean of the clusters

    return new_centroids,clusters



def kmeans(data, k):
   
    centroids = data[np.random.randint(data.shape[0], size=k)]   # initialize k centroids randomly
    
    while True:
       
        new_centroids,clusters = getcentroids(data,k,centroids) 
    
        if np.isin(new_centroids,centroids).all():
            break
        centroids = np.copy(new_centroids) #new centroid value will be copied for further comparisons
        
    return centroids, clusters

image_path = "bird-thumbnail_old_image.jpg" #providing the image path
image = cv2.imread(image_path)
h,w,d = image.shape 

img_flattened=np.reshape(image,(h*w,d)) #flattening the image from 2D to 1D
img_flattened = img_flattened.astype('int64') 




plt.imshow(np.asarray(image)) #plotting the image
plt.show()


centroids, clusters = kmeans(img_flattened,3) #func call

clusters = np.reshape(clusters,(h,w)) #resizing the output to image size

plt.imshow(np.asarray(clusters),cmap='jet',alpha=0.5)
plt.show()

