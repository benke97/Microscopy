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
from write_PSLG import write_PSLG
from material import Material

im = skimage.io.imread("test.tif")
refined_positions = get_column_positions(im,0.1,0.2,0.009)
x = refined_positions[:,1]
y = refined_positions[:,0]
number_of_phases = 2 
fig, ax = plt.subplots()
pts = ax.scatter(x, y)
number_of_phases = 2

polygon_select1 = SelectFromCollection(ax, pts)
plt.axis('square')
plt.show()
polygon_select1.disconnect()
#print('\nSelected points:')
#print(polygon_select1.xys[polygon_select1.ind])
platinum = polygon_select1.xys[polygon_select1.ind]
fig, ax = plt.subplots()
pts = ax.scatter(x, y)
polygon_select2 = SelectFromCollection(ax, pts)
plt.axis('square')
plt.show()
polygon_select2.disconnect()
#print('\nSelected points:')
#print(polygon_select2.xys[polygon_select2.ind])
cerium = polygon_select2.xys[polygon_select2.ind]
phases = [platinum,cerium]
names = ['platinum','cerium']
triangulations = []
for i in range(number_of_phases):
#----------------------Click on edges in order--------------------------------------
    fig, ax = plt.subplots()
    coll = ax.scatter(phases[i][:,0], phases[i][:,1], color=["blue"]*len(phases[i]), picker = 5, s=[50]*len(phases[i]))
    edge_loop_idx = []
    start_pos = 0


    def on_pick(event):
        global start_pos
        print(start_pos,event.ind)
        if np.array_equal(event.ind,start_pos):
            plt.close(fig)
        
        if not start_pos:
            print("test")
            start_pos = event.ind
        print(phases[i][event.ind], "clicked")

        if np.array_equal(coll._facecolors[event.ind,:], np.array([[0,0,1,1]])):
            edge_loop_idx.append(event.ind)
            coll._facecolors[event.ind,:] = (1,0,0,1)
            coll._edgecolors[event.ind,:] = (1,0,0,1)
        
        elif np.array_equal(coll._facecolors[event.ind,:], np.array([[1,0,0,1]])):
            coll._facecolors[event.ind,:] = (0,0,1,1)
            coll._edgecolors[event.ind,:] = (0,0,1,1)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
    fig.canvas.mpl_disconnect(coll)
    plt.close(fig)
    #print(edge_loop_idx)
    #print(edge_loop_idx[1])
    #----------------------------Generate PSLG-----------------------------------

    write_PSLG(names[i],phases[i],edge_loop_idx)

#open and read the file after the appending:
    #f = open(str(names[i])+".poly", "r")
#print(f.read())

#---------Show constrained delaunay---------------------
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names[i]))
    t = tr.triangulate(particle, 'pne')
    triangulations.append(t)

    tr.compare(plt, particle, t)
    plt.show()



platin = Material(triangulations[0])
ceri = Material(triangulations[1])

print(platin.trig_strain)




Xboi= np.array(platin.vertices)
Xbaoo = np.array(platin.ideal_vertices)
plt.scatter(Xboi[:,0], Xboi[:,1], c ="b")
plt.scatter(Xbaoo[:,0],Xbaoo[:,1],color="r") 
# To show the plot
plt.show()
#print(platin.connections)

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
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'plasma')
#tr.plot(ax,**t)
ax.quiver(x_pos,y_pos,x_dir,y_dir,angles='xy', scale_units='xy', scale=1)
plt.show()




V = np.array(platin.center_neighborhood_vectors)
origin = np.repeat(np.array([[np.array(platin.vertices[platin.central_vertex])[0]],[np.array(platin.vertices[platin.central_vertex])[1]]]),np.size(V,axis=0),axis=1)

fig, ax = plt.subplots()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
tr.plot(ax,**triangulations[0])
tr.plot(ax,**triangulations[1])
ax.quiver(x_pos,y_pos,x_dir,y_dir,angles='xy', scale_units='xy', scale=1, color='m')
#ax.quiver(*origin, V[:,0],V[:,1], color=['r','b','g','k','y','c'],angles='xy', scale_units='xy', scale=1)
ax.tripcolor(np.array(platin.vertices)[:,0], np.array(platin.vertices)[:,1], np.array(platin.triangles),facecolors=np.array(platin.trig_strain), cmap='coolwarm',alpha=0.2, edgecolors='k')
ax.tripcolor(np.array(ceri.vertices)[:,0], np.array(ceri.vertices)[:,1], np.array(ceri.triangles),facecolors=np.array(ceri.trig_strain), cmap='coolwarm',alpha=0.2, edgecolors='k')

print(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0))
print(np.array(platin.triangles))
print(np.array(platin.trig_strain))


plt.show()

#----------------------------------------------------------------------------


img = plt.imread("test.tif")
implot = plt.imshow(img)
plt.scatter(platinum[:,1], platinum[:,0], s=30, c='r', marker="o", label='Pt')
plt.scatter(cerium[:,1],cerium[:,0], s=30, c='b', marker="o", label='Ce')
plt.legend(loc='upper left')
#plt.axis('square')
plt.show()
