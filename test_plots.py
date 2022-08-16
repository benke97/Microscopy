from cmath import inf
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from tarfile import TarError
from turtle import color, shape
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
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

im = skimage.io.imread("test.tif")
triangulations = []
names = ['boitest','cerium']
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

im = skimage.io.imread("test3.tif")
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

print(np.shape(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,0]),np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,1])
print(np.shape(np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0)))
print(np.shape(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))



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

print(np.shape(np.array((platin.voronoi_verts))))
print(platin.voronoi_segs)
print(platin.segments)

from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(platin.vertices)
#fig = voronoi_plot_2d(vor)
print(platin.edges)
print(platin.segments)
#print(platin.voronoi_vertices)
#fig, ax = plt.subplots()
#voronoi_plot_2d(vor)
#tr.plot(ax,**triangulations[0])
#plot lines
#print(np.shape(np.array(platin.triangles)))
#print(np.shape(np.array(platin.points)),len(platin.vertices),platin.points)
#print(np.array(platin.vertices)[platin.triangles[0]], platin.voronoi_verts[0])

#DONT DELETE THIS WAS HARD!
#a = [np.vstack((np.array(platin.voronoi_verts)[seg])) for seg in platin.voronoi_segs]
#line_segments = LineCollection(a, linewidths=2,
#                               colors='b', linestyle='solid')
#ax.add_collection(line_segments)
#im_data = im.T
#imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
#ax.scatter(np.array(platin.voronoi_verts)[:,0],np.array(platin.voronoi_verts)[:,1], s=100, c='k',marker='o')
#ax.scatter(np.array(platin.vertices)[:,0],np.array(platin.vertices)[:,1], s=60, c='r')
#plt.axis('square')
#plt.show()

print(platin.voronoi_edges)
fig,ax = plt.subplots()

boi = ax.tripcolor(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,0],np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,1],
np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0),facecolors=np.append(np.array(platin.trig_strain),
np.array(ceri.trig_strain)), cmap='coolwarm',alpha=0.7, edgecolors='k')

a = [np.vstack((np.array(ceri.voronoi_verts)[seg])) for seg in ceri.voronoi_segs]
line_segments = LineCollection(a, linewidths=2,
                               colors='k', linestyle='solid')
ax.add_collection(line_segments)
a = [np.vstack((np.array(platin.voronoi_verts)[seg])) for seg in platin.voronoi_segs]
line_segments2 = LineCollection(a, linewidths=2,
                               colors='k', linestyle='solid')
ax.add_collection(line_segments2)
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
#ax.scatter(np.array(ceri.ideal_vertices)[:,0],np.array(ceri.ideal_vertices)[:,1], s=60, c='c',marker='o')
#ax.scatter(np.array(ceri.points)[:,0],np.array(ceri.points)[:,1], s=60, c='g',marker='o')
#ax.scatter(np.array(ceri.voronoi_verts)[:,0],np.array(ceri.voronoi_verts)[:,1], s=100, c='k',marker='o')
#ax.scatter(np.array(ceri.vertices)[:,0],np.array(ceri.vertices)[:,1], s=60, c='r')
#plt.axis('square')
plt.show()



fig, ax = plt.subplots()
patches = []
for cell_nr in range(len(platin.voronoi_bulk)):
    polygon = Polygon(np.array(platin.voronoi_verts)[np.array(platin.voronoi_cells)[platin.voronoi_bulk][cell_nr]], True)
    patches.append(polygon)

for cell_nr in range(len(ceri.voronoi_bulk)):
    polygon = Polygon(np.array(ceri.voronoi_verts)[np.array(ceri.voronoi_cells)[ceri.voronoi_bulk][cell_nr]], True)
    patches.append(polygon)


p = PatchCollection(patches,alpha = 0.7,cmap='coolwarm')
print(np.append(np.array(platin.voronoi_rel_size)[platin.voronoi_bulk],np.array(platin.voronoi_rel_size)[platin.voronoi_bulk]))
colors = np.append(np.array(platin.voronoi_rel_size)[platin.voronoi_bulk],np.array(ceri.voronoi_rel_size)[ceri.voronoi_bulk])
p.set_array(colors,)
boi = ax.add_collection(p)
fig.colorbar(boi)

a = [np.vstack((np.array(ceri.voronoi_verts)[seg])) for seg in ceri.voronoi_segs]
line_segments2 = LineCollection(a, linewidths=2,
                               colors='k', linestyle='solid')
ax.add_collection(line_segments2)
a = [np.vstack((np.array(platin.voronoi_verts)[seg])) for seg in platin.voronoi_segs]
line_segments = LineCollection(a, linewidths=2,
                               colors='k', linestyle='solid')
ax.add_collection(line_segments)

ax.autoscale_view()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
plt.show()



im3 = skimage.io.imread("test3.tif")
im_data = im3.T

fig, ax = plt.subplots()
patches = []
for cell_nr in range(len(platin.voronoi_bulk)):
    polygon = Polygon(np.array(platin.voronoi_verts)[np.array(platin.voronoi_cells)[platin.voronoi_bulk][cell_nr]], True)
    patches.append(polygon)

p = PatchCollection(patches,alpha = 0.7,cmap='coolwarm')
colors = np.array(platin.voronoi_rel_size)[platin.voronoi_bulk]
p.set_array(colors,)
p.set_clim([-15,15])
ax.add_collection(p)
fig.colorbar(p)

a = [np.vstack((np.array(platin.voronoi_verts)[seg])) for seg in platin.voronoi_segs]
line_segments = LineCollection(a, linewidths=2,
                               colors='k', linestyle='solid')
ax.add_collection(line_segments)

ax.autoscale_view()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
plt.show()

from Plotter import Plotter

a = Plotter([platin,ceri],['platinum','cerium'],im3)
a.plot_voronoi(['platinum'],strain=True)
#a.plot_voronoi(['platinum','cerium'])

#a.plot_displacement_map(['platinum','cerium'])
a.plot_delaunay(['platinum'],strain=True)
#print(np.amax(np.array(platin.number_of_connections)))
#print(platin.number_of_connections[platin.central_vertex])
#a.plot_delaunay(['platinum','cerium'])
#a.plot_delaunay(['cerium','platinum'],strain=True)
#a.plot_voronoi(['platinum','cerium'])