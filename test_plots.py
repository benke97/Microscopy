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
from matplotlib.collections import LineCollection

im = skimage.io.imread("test.tif")
triangulations = []
names = ['platinum','cerium']
for n in range(len(names)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names[n]))
    t = tr.triangulate(particle, 'pne')
    triangulations.append(t)



ceri = Material(triangulations[1])
platin = Material(triangulations[0])

#print(platin.trig_strain)
#fig, ax = plt.subplots()
#im_data = im.T



#plt.figure()
#plt.gca().set_aspect('equal')
#plt.tripcolor(np.array(platin.vertices)[:,0], np.array(platin.vertices)[:,1], np.array(platin.triangles), facecolors=np.array(platin.trig_strain),cmap='coolwarm',alpha=0.4, edgecolors='k')
#plt.imshow(im_data,cmap='Greys')
#plt.colorbar()
#plt.title('tripcolor')

#plt.show()




#Xboi= np.array(platin.vertices)
#Xbaoo = np.array(platin.ideal_vertices)
#plt.scatter(Xboi[:,0], Xboi[:,1], c ="b")
#plt.scatter(Xbaoo[:,0],Xbaoo[:,1],color="r") 
# To show the plot
#plt.show()
#print(platin.connections)


V = np.array(platin.vertex_displacements[0])
x,y = -V.T #IMPORTANT MINUS SIGN
x_dir = x.tolist()
y_dir = y.tolist()
x,y = np.array(platin.vertices).T
x_pos = x.tolist()
y_pos = y.tolist()

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

boi = ax.tripcolor(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,0],np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,1],
np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0),facecolors=np.append(np.array(platin.trig_strain),
np.array(ceri.trig_strain)), cmap='coolwarm',alpha=0.5, edgecolors='k')
fig.colorbar(boi)
plt.show()

#----------------------------------------------------------------------------

img = plt.imread("test.tif")
implot = plt.imshow(img)
plt.scatter(np.array(platin.vertices)[:,1], np.array(platin.vertices)[:,0], s=30, c='r', marker="o", label='Pt')
plt.scatter(np.array(ceri.vertices)[:,1],np.array(ceri.vertices)[:,0], s=30, c='b', marker="o", label='Ce')
plt.legend(loc='upper left')
#plt.axis('square')
plt.show()



from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(platin.vertices)
#fig = voronoi_plot_2d(vor)
print(platin.edges)
print(platin.segments)
#print(platin.voronoi_vertices)
fig, ax = plt.subplots()
#voronoi_plot_2d(vor)
tr.plot(ax,**triangulations[0])
#plot lines
print(np.shape(np.array(platin.triangles)))
#print(np.array(platin.vertices)[platin.triangles[0]], platin.voronoi_verts[0])
#DONT DELETE THIS WAS HARD!

a = [np.vstack((np.array(platin.voronoi_verts)[seg])) for seg in platin.voronoi_segs]
line_segments = LineCollection(a, linewidths=2,
                               colors='b', linestyle='solid')
ax.add_collection(line_segments)

#ax.scatter(np.array(platin.vertices)[:,0],np.array(platin.vertices)[:,1], s=30, c='b')
ax.scatter(np.array(platin.voronoi_verts)[:,0],np.array(platin.voronoi_verts)[:,1], s=50, c='k',marker= 'x')
#plt.axis('square')
plt.show()
