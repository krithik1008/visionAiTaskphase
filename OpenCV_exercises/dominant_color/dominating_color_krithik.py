import cv2 as cv
from  sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections
import statistics as st

img=cv.imread('6.jpg')
img=cv.resize(img,(400,400),interpolation = cv.INTER_AREA)


cv.imshow('img',img)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.waitKey(0)
cv.destroyAllWindows()


#clustering ppixels into groups using Kmeans clustering
img1 = img.reshape((img.shape[0] * img.shape[1], 3)) 
km = KMeans(n_clusters = 5,random_state=48)
km.fit(img1)

pred= km.predict(img1)

print(km.cluster_centers_)

#finding the most frequent cluster
dom=st.mode(pred)
colors=km.cluster_centers_

#assigning most dominant  color
dcolor = colors[dom] 
print("Most dominating color is:",dcolor)

count=collections.Counter(pred)
#plotting most dominant colors according to the clustering algo
barp=plt.bar((count.keys()),(count.values()))
barp[0].set_color(colors[0]/255)
barp[1].set_color(colors[1]/255)
barp[2].set_color(colors[2]/255)
barp[3].set_color(colors[3]/255)
barp[4].set_color(colors[4]/255)
plt.xlabel('Color')
plt.ylabel('No. of pixels')
plt.show()