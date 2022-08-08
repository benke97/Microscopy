from cmath import inf
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from tarfile import TarError
from turtle import shape
import stemtool as st
import cv2
import skimage
import numpy as np
from skimage import io
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from get_column_positions import get_column_positions
from Polygon_selector import SelectFromCollection
import scipy.optimize as opt
import triangle as trrr
from sklearn.neighbors import NearestNeighbors
import triangle as tr
import os 
import math 
from scipy.spatial import distance
from material import Material

im = skimage.io.imread("test.tif")
triangulations = []
names = ['platinum','cerium']
for n in range(len(names)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names[n]))
    t = tr.triangulate(particle, 'pne')
    triangulations.append(t)



platin = Material(triangulations[0])
ceri = Material(triangulations[1])

print(platin.trig_strain)
#fig, ax = plt.subplots()
#im_data = im.T



#plt.figure()
#plt.gca().set_aspect('equal')
#plt.tripcolor(np.array(platin.vertices)[:,0], np.array(platin.vertices)[:,1], np.array(platin.triangles), facecolors=np.array(platin.trig_strain),cmap='coolwarm',alpha=0.4, edgecolors='k')
#plt.imshow(im_data,cmap='Greys')
#plt.colorbar()
#plt.title('tripcolor')

#plt.show()




Xboi= np.array(platin.vertices)
Xbaoo = np.array(platin.ideal_vertices)
plt.scatter(Xboi[:,0], Xboi[:,1], c ="b")
plt.scatter(Xbaoo[:,0],Xbaoo[:,1],color="r") 
# To show the plot
plt.show()
#print(platin.connections)
print(platin.vertices)
print('\n')
print(platin.vertex_displacements[0])
print(platin.ideal_trig_areas)
print(platin.triangle_areas)
#print(platin.connection_classes)
#print('\n')
#print(platin.edges)
#print('\n')
#print(platin.edge_classes)
#print(platin.number_of_connections)
#print(platin.connections)
#print(platin.central_vertex)
#print(platin.segment_areas)
V = np.array(platin.vertex_displacements[0])
x,y = -V.T #IMPORTANT MINUS SIGN
x_dir = x.tolist()
y_dir = y.tolist()
x,y = np.array(platin.vertices).T
x_pos = x.tolist()
y_pos = y.tolist()
print(len(x))
print(len(y))
print(len(x_pos))
print(len(y_pos))
#print(origina)
#print(np.array(platin.vertices))
fig, ax = plt.subplots()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
#tr.plot(ax,**t)
ax.quiver(x_pos,y_pos,x_dir,y_dir,angles='xy', scale_units='xy', scale=1,color='b')
plt.show()




V = np.array(platin.center_neighborhood_vectors)
origin = np.repeat(np.array([[np.array(platin.vertices[platin.central_vertex])[0]],[np.array(platin.vertices[platin.central_vertex])[1]]]),np.size(V,axis=0),axis=1)

fig, ax = plt.subplots()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
tr.plot(ax,**triangulations[0])
tr.plot(ax,**triangulations[1])
#ax.quiver(x_pos,y_pos,x_dir,y_dir,angles='xy', scale_units='xy', scale=1, color='m')
#ax.quiver(*origin, V[:,0],V[:,1], color=['r','b','g','k','y','c'],angles='xy', scale_units='xy', scale=1)
#ax.tripcolor(np.array(platin.vertices)[:,0], np.array(platin.vertices)[:,1], np.array(platin.triangles),facecolors=np.array(platin.trig_strain), cmap='coolwarm',alpha=0.2, edgecolors='k')
#ax.tripcolor(np.array(ceri.vertices)[:,0], np.array(ceri.vertices)[:,1], np.array(ceri.triangles),facecolors=np.array(ceri.trig_strain), cmap='coolwarm',alpha=0.2, edgecolors='k')

print(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0))
print(np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0))
#print(np.amax(np.array(platin.triangles)))
#print(np.amin(np.array(ceri.triangles)+36))
#print(np.array(platin.triangles))
#print(np.array(ceri.triangles)+35)
print(np.size(np.array(platin.trig_strain)))
print(np.size(np.array(platin.triangles)))
print(np.size(np.array(ceri.trig_strain)))
print(np.size(np.array(ceri.triangles)))
print(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain)))

boi = ax.tripcolor(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,0],np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,1],
np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0),facecolors=np.append(np.array(platin.trig_strain),
np.array(ceri.trig_strain)), cmap='coolwarm',alpha=0.7, edgecolors='k')
fig.colorbar(boi)
print(np.amax(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
print(np.amin(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
plt.show()

#----------------------------------------------------------------------------

img = plt.imread("test.tif")
implot = plt.imshow(img)
plt.scatter(np.array(platin.vertices)[:,1], np.array(platin.vertices)[:,0], s=30, c='r', marker="o", label='Pt')
plt.scatter(np.array(ceri.vertices)[:,1],np.array(ceri.vertices)[:,0], s=30, c='b', marker="o", label='Ce')
plt.legend(loc='upper left')
#plt.axis('square')
plt.show()
