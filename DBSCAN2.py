import numpy as np
import cv2
import matplotlib.pyplot as plt

def dbscan(data, eps, min_samples):
    # initialize all data points as unvisited
    visited = np.zeros(data.shape[0])
    clusters = np.zeros(data.shape[0]) 
    clusters.fill(-1) #filling -1
    cluster_count=0
    
    # iterate through all unvisited data points
    for i in range(data.shape[0]):
        if not visited[i]:
            visited[i] = 1
            neighborhood = get_neighbpoints(data, i, eps) #to check the neighbourhood points

            # if the number of points in the neighborhood is less than min_samples, mark it as noise
            if len(neighborhood) < min_samples:
                visited[i] = -1 #noise
            else:
                # otherwise start a new cluster
               
                
                clusters[i]=cluster_count
                visited[i] = 1

                # expand the cluster by checking all points in the neighborhood
                for point in neighborhood:
                    if not visited[point]:
                        visited[point] = 1
                        new_neighborhood = get_neighbpoints(data, point, eps)

                        # if the number of points in the new neighborhood is greater than or equal to min_samples, add it to the cluster
                        if len(new_neighborhood) >= min_samples:
                            #neighborhood += new_neighborhood
                            neighborhood = np.append(neighborhood,new_neighborhood)
                            
                    if visited[point] != 1:
                        clusters[point]=cluster_count
                        visited[point] = 1
                        
                cluster_count+=1

    return clusters

def get_neighbpoints(data, point, eps):
    # return all data points within a distance of eps from the given point
    return np.where(np.linalg.norm(data - data[point], axis=1) <= eps)[0] #eucledian distance to see if the points is less than epsilon




image_path = "bird-thumbnail_old_image.jpg" #providing the image
image = cv2.imread(image_path)
h,w,d = image.shape

img_flattened=np.reshape(image,(h*w,d))  #flattening the image
clusters =dbscan(img_flattened, 50, 3)
clusters = np.reshape(clusters,(h,w))  #reshaping the image

plt.imshow(np.asarray(image))
plt.show()


plt.imshow(np.asarray(clusters),cmap='jet',alpha=0.5)
plt.show()
