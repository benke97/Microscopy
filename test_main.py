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

im = skimage.io.imread("test.tif")
refined_positions = get_column_positions(im,0.1,0.2)
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

    f = open(str(names[i]) + ".poly", "w")
    #-----
    f.write("# " + str(names[i]) + ".poly" + '\n')
    f.write("#" + '\n')
    f.write("# A" + str(names[i]) + "particle with X points in 2D, no attributes, one boundary marker." + '\n')
    f.write(str(len(phases[i])) + " 2" + " 0" + " 1" + '\n')
    f.write("# Perimeter" + '\n')

    for j in phases[i]:
        atom_idx = np.where(np.all(phases[i]==j,axis=1))
        print(atom_idx[0])
        atom_idx = atom_idx[0]
        if atom_idx in edge_loop_idx:
            f.write(str(int(atom_idx+1)) + " " + str(phases[i][atom_idx][0,0]) + " " + str(phases[i][atom_idx][0,1]) + " 2" + '\n')
        else:
            f.write(str(int(atom_idx+1)) + " " + str(phases[i][atom_idx][0,0]) + " " + str(phases[i][atom_idx][0,1]) + " 0" + '\n')

    #for i in edge_loop_idx:
        #print(platinum[i])
        #f.write(str(edge_loop_idx.index(i)+1) + " " + str(platinum[i][0,0]) + " " + str(platinum[i][0,1]) + " 2" + '\n')
    f.write('\n')
    #-----
    f.write("# X segments, each with boundary marker." + '\n')
    f.write(str(len(edge_loop_idx)) + " 1" + '\n')
    f.write("# Perimeter" + '\n')

    for j in edge_loop_idx:
        vertex_idx = edge_loop_idx.index(j)
        if vertex_idx+1 == len(edge_loop_idx):
            next_value = edge_loop_idx[0]
        else:
            print(vertex_idx,len(edge_loop_idx))
            print(int(vertex_idx+1))
            next_value = edge_loop_idx[int(vertex_idx+1)] 
        f.write(str(int(vertex_idx+1)) + " " + str(int(j+1)) + " " + str(int(next_value+1)) + " 2" + '\n')





    #for i in edge_loop_idx:
    #    if edge_loop_idx.index(i) + 1 < len(edge_loop_idx):
    #        f.write(str(edge_loop_idx.index(i)+1) + " " + str(edge_loop_idx.index(i)+1) + " " + str((edge_loop_idx.index(i)+2)%(len(edge_loop_idx)+1)) + " 2" + '\n')
    #    else:
    #        f.write(str(edge_loop_idx.index(i)+1) + " " + str(edge_loop_idx.index(i)+1) + " " + " 1" + " 2" + '\n')
    f.write('\n')
    #-----
    f.write("# No holes" + '\n')
    f.write("0")

    f.close()

#open and read the file after the appending:
    #f = open(str(names[i])+".poly", "r")
#print(f.read())

#---------Show constrained delaunay---------------------
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names[i]))
    t = tr.triangulate(particle, 'pne')
    triangulations.append(t)
#print(t['neighbors'].tolist())
#print(t['vertices'].tolist())
#print(t['triangles'].tolist())
#print(t['edges'].tolist())
#print(t['segments'].tolist())
#print(t['regions'].tolist())
#print(t['triangle_attributes'].tolist())
    tr.compare(plt, particle, t)
    plt.show()

class Material:

    def __init__(self,triangle_object):
        self.number_of_connections = []
        self.connections = []
        self.connection_classes = []
        self.triangle_areas = []
        self.ideal_vertices = []
        self.ideal_trig_areas = []
        self.edge_lengths = []
        self.edge_classes = []
        self.center_neighborhood_vectors = []
        self.vertex_displacements = []
        self.trig_strain = []
        self.central_vertex = 0
        self.primitive_vectors = np.zeros((2,2))
        self.neighbors = triangle_object['neighbors'].tolist()
        self.vertices = triangle_object['vertices'].tolist() 
        self.triangles = triangle_object['triangles'].tolist()
        self.edges = triangle_object['edges'].tolist()
        self.set_number_of_connections()  
        self.set_connections()
        self.set_central_vertex()
        self.set_center_neighborhood_vectors()
        self.calculate_edge_lengths()
        self.calculate_triangle_areas()
        self.classify_segments()
        self.set_edge_class()
        self.calculate_vertex_displacement()
        self.set_ideal_vertices()
        self.calculate_ideal_triangles()
        self.calc_trig_strain()

    def get_length(self,idx_1,idx_2,*args):
        if 'ideal' in args:
            a = self.ideal_vertices[idx_1]
            b = self.ideal_vertices[idx_2]
        else:
            a = self.vertices[idx_1]
            b = self.vertices[idx_2]
        return math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))    
    
    def calc_trig_area(self,a,b,c):
        s = (a+b+c)/2
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        return area

    def calculate_triangle_areas(self):
        #Heron's formula
        for trig in self.triangles:
            a = self.get_length(trig[0],trig[1])
            b = self.get_length(trig[1],trig[2])
            c = self.get_length(trig[2],trig[0])
            self.triangle_areas.append(self.calc_trig_area(a,b,c))

    def calculate_edge_lengths(self):
        for edge in self.edges:
            self.edge_lengths.append(self.get_length(edge[0],edge[1]))
        print('hello')

    def set_number_of_connections(self):
        for i in range(len(self.vertices)):
            self.number_of_connections.append(sum(vertex_pair.count(i) for vertex_pair in self.edges))
            #print(sum(vertex_pair.count(i) for vertex_pair in self.edges))

    def set_central_vertex(self):
        vertex_array = np.array(self.vertices)
        centroid = np.array([np.sum(vertex_array[:,0])/np.size(vertex_array[:,0]),np.sum(vertex_array[:,1])/np.size(vertex_array[:,1])])
        
        centroid_topped_vertex_array = np.concatenate(([centroid],vertex_array),axis=0)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(centroid_topped_vertex_array)
        distances, indices = nbrs.kneighbors(centroid_topped_vertex_array)
        index_nn = indices[0,1]
        self.central_vertex = index_nn-1

    def set_connections(self):

        for i in range(len(self.vertices)):
            connection_list = []
            for [x,y] in self.edges:
                if i == x: 
                    connection_list.append(y)
                if i == y:
                    connection_list.append(x)
            self.connections.append(connection_list)

    def set_center_neighborhood_vectors(self):
         for neighbor in self.connections[self.central_vertex]:
            x = self.vertices[neighbor][0]-self.vertices[self.central_vertex][0]
            y = self.vertices[neighbor][1]-self.vertices[self.central_vertex][1]
            self.center_neighborhood_vectors.append([x,y])  

    def classify_segments(self):
        duplicate_bool = 0
        for i in range(len(self.vertices)):
            #för varje punkt, get neighbors, för varje neighbor, lägg vektorn (granne - i) i lista, kör cdist med neighborhood_vectors, hitta index av min(row)
            neighbor_vectors = []
            for neighbor in self.connections[i]:
                    x = self.vertices[neighbor][0]-self.vertices[i][0]
                    y = self.vertices[neighbor][1]-self.vertices[i][1]
                    neighbor_vectors.append([x,y])
            boi = np.array(distance.cdist(neighbor_vectors,self.center_neighborhood_vectors, 'cosine')) #rows columns
            #print(boi)
            classifier = np.argmin(boi, axis=1)


            print(classifier)
            if len(classifier) != len(set(classifier)): #if duplicate find best matching of the two and set ignore segment.
                i = 0
                j = 0
                cosine_val = -inf
                for cls in classifier:
                    print(boi[i,cls])
                    if boi[i,cls] > cosine_val:
                        cosine_val = boi[i,cls]
                        j = i
                    i+=1
                duplicate_bool = 1
                classifier[j] = 99 #ignore 
                #print(classifier)
                print('duplicate detected', i, self.vertices[i])
            self.connection_classes.append(classifier.tolist())
        if not duplicate_bool:
            print('no duplicates detected')
            
    def set_edge_class(self):
        i = 0
        edge_class_list = [[] for _ in range(len(self.edges))]
        #print(edge_class_list)
        for connection_list in self.connections:
            j = 0
            for connection in connection_list:
                if [i,connection] in self.edges:    
                    edge_idx = self.edges.index([i,connection])
                elif [connection,i] in self.edges:
                    edge_idx = self.edges.index([connection,i])
                #print(len(self.edges),edge_idx)
                edge_class_list[edge_idx].append(self.connection_classes[i][j])
                j +=1
            i += 1        
        self.edge_classes = edge_class_list
    
    def get_edge_index(self,vertex_idx1,vertex_idx2):
        edge_idx = 0
        if [vertex_idx1,vertex_idx2] in self.edges or [vertex_idx2,vertex_idx1] in self.edges:
            if [vertex_idx1,vertex_idx2] in self.edges:    
                edge_idx = self.edges.index([vertex_idx1,vertex_idx2])
            elif [vertex_idx2,vertex_idx1] in self.edges:
                edge_idx = self.edges.index([vertex_idx2,vertex_idx1])
            return edge_idx
        else:
            raise ValueError('no segment between indices' + str(vertex_idx1) + 'and' + str(vertex_idx2))


    def calculate_vertex_displacement(self):
        displacement_list = []
        for vertex in range(len(self.vertices)):
            ideal_vector_list = []
            vector_list = []
  
            for neighbor in self.connections[vertex]:
                edge_idx = self.get_edge_index(vertex,neighbor)
                [class1,class2]= [self.edge_classes[edge_idx][0],self.edge_classes[edge_idx][1]] #each segment classified from both directions
                if class1 == 99 or class2 == 99:
                    #ignore this line segment in the displacement calculations since it has been flagged that
                    #it does not share resemblance with any of the reference vectors
                    print('ignored segment')
                else:
                    #mean vector 
                    u = [(self.center_neighborhood_vectors[class1][0]-self.center_neighborhood_vectors[class2][0])/2,
                    (self.center_neighborhood_vectors[class1][1]-self.center_neighborhood_vectors[class2][1])/2]
                    

                    v = [(self.vertices[neighbor][0]-self.vertices[vertex][0]),
                    (self.vertices[neighbor][1]-self.vertices[vertex][1])]
                    #print(distance.cosine(v,u),v,u)
                    if distance.cosine(v,u) <= 1/math.sqrt(2):
                        ideal_vector_list.append(u)
                        vector_list.append(v)
                        #i +=1
                    else:
                        u = [-u[0],-u[1]]
                        ideal_vector_list.append(u)
                        vector_list.append(v)
                        #j+=1
            ideal_vecs=np.array(ideal_vector_list)
            real_vecs=np.array(vector_list)
            displacements = np.sum(np.subtract(real_vecs,ideal_vecs),axis=0)/np.shape(real_vecs)[1]
            #print(ideal_vecs,real_vecs)
            displacement_list.append(displacements.tolist())
        self.vertex_displacements.append(displacement_list)
    
    def set_ideal_vertices(self):
        for vertex in range(len(self.vertices)):
            ideal_pos = np.array(self.vertices[vertex])+np.array(self.vertex_displacements[0][vertex])
            self.ideal_vertices.append(ideal_pos.tolist())  

    def calculate_ideal_triangles(self): #this is not correct
        for trig in self.triangles:
            a = self.get_length(trig[0],trig[1],'ideal')
            b = self.get_length(trig[1],trig[2],'ideal')
            c = self.get_length(trig[2],trig[0],'ideal')
            self.ideal_trig_areas.append(self.calc_trig_area(a,b,c))

        print('hello')

    def calc_trig_strain(self):
        i = 0
        for trig in self.triangles:
            a_idx = self.get_edge_index(trig[0],trig[1])
            b_idx = self.get_edge_index(trig[1],trig[2])
            c_idx = self.get_edge_index(trig[2],trig[0])
            class1a,class2a = self.edge_classes[a_idx]
            class1b,class2b = self.edge_classes[b_idx]
            class1c,class2c = self.edge_classes[c_idx]
            
            if class1a == 99 or class2a == 99:
                a_length = self.get_length(trig[0],trig[1])
            else:
                #a_length = (np.linalg.norm(np.array(self.center_neighborhood_vectors[class1a]))+np.absolute(np.array(self.center_neighborhood_vectors[class2a])))/2
                a_length = (math.sqrt(pow(self.center_neighborhood_vectors[class1a][0],2)+pow(self.center_neighborhood_vectors[class1a][1],2))
                            + math.sqrt(pow(self.center_neighborhood_vectors[class2a][0],2)+pow(self.center_neighborhood_vectors[class2a][1],2)))/2
            if class1b == 99 or class2b == 99:
                b_length = self.get_length(trig[1],trig[2])
            else:
                #b_length = (np.linalg.norm(np.array(self.center_neighborhood_vectors[class1b]))+np.absolute(np.array(self.center_neighborhood_vectors[class2b])))/2
                b_length = (math.sqrt(pow(self.center_neighborhood_vectors[class1b][0],2)+pow(self.center_neighborhood_vectors[class1b][1],2))
                            + math.sqrt(pow(self.center_neighborhood_vectors[class2b][0],2)+pow(self.center_neighborhood_vectors[class2b][1],2)))/2            
            if class1b == 99 or class2b == 99:
                c_length = self.get_length(trig[2],trig[0])
            else:
                #c_length = (np.linalg.norm(np.array(self.center_neighborhood_vectors[class1c]))+np.absolute(np.array(self.center_neighborhood_vectors[class2c])))/2
                c_length = (math.sqrt(pow(self.center_neighborhood_vectors[class1c][0],2)+pow(self.center_neighborhood_vectors[class1c][1],2))
                            + math.sqrt(pow(self.center_neighborhood_vectors[class2c][0],2)+pow(self.center_neighborhood_vectors[class2c][1],2)))/2


            print(self.center_neighborhood_vectors[class1a])
            print(np.array(self.center_neighborhood_vectors[class1a]))
            print(np.absolute(np.array(self.center_neighborhood_vectors[class1a])))
            print(a_length,b_length,c_length)
            ideal_area = self.calc_trig_area(a_length,b_length,c_length)
            strain_percent = (1-self.triangle_areas[i]/ideal_area)*100
            self.trig_strain.append(strain_percent)
            i +=1


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
PCM=ax.get_children()[2]
plt.colorbar(PCM,ax=ax)
plt.show()

#----------------------------------------------------------------------------


img = plt.imread("test.tif")
implot = plt.imshow(img)
plt.scatter(platinum[:,1], platinum[:,0], s=30, c='r', marker="o", label='Pt')
plt.scatter(cerium[:,1],cerium[:,0], s=30, c='b', marker="o", label='Ce')
plt.legend(loc='upper left')
#plt.axis('square')
plt.show()

#babbo = np.array([np.sum(platinum[:,0])/np.size(platinum[:,0]),np.sum(platinum[:,1])/np.size(platinum[:,1])])
#print(babbo)
#testerrr = np.concatenate(([babbo],platinum),axis=0)
#print(testerrr)
#print(babbo)

#plt.scatter(platinum[:,0], platinum[:,1], s=30, c='r', marker="o")
#plt.scatter(babbo[0],babbo[1],s=40, c='b', marker='x')
#plt.show()

#nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(testerrr)
#distances, indices = nbrs.kneighbors(testerrr)
#print(indices)
#print(distances)
#index_nn = indices[0,1]

#nn = platinum[index_nn-1,:]
#print(nn)
#print(platin.vertices[platin.central_vertex])
#print(babbo)
#vertex_array = np.array(platin.vertices)
#centroidboi = np.array([np.sum(vertex_array[:,0])/np.size(vertex_array[:,0]),np.sum(vertex_array[:,1])/np.size(vertex_array[:,1])])
#print(centroidboi)

#plt.scatter(platinum[:,0], platinum[:,1], s=30, c='r', marker="o")
#plt.scatter(babbo[0],babbo[1],s=40, c='b', marker='x')
#plt.scatter(platin.vertices[platin.central_vertex][0],platin.vertices[platin.central_vertex][1],s=50, c='k', marker='o')
#plt.scatter(nn[0],nn[1],s=50, c='k', marker='o')
#plt.axis('square')
#plt.show()            

#fig, ax = plt.subplots()
#im_data = im.T
#imaa = ax.imshow(im_data,origin = 'lower',cmap = 'plasma')
#tr.plot(ax,**t)
#plt.show()

#def get_primitive_vectors(lattice_point)





#from scipy.spatial import Delaunay
#tri = Delaunay(np.array(platinum),qhull_options="Q14")
#plt.triplot(platinum[:,1], platinum[:,0], tri.simplices)
#plt.plot(platinum[:,1], platinum[:,0], 'o')
#plt.show()
#remaining_peaks = np.delete(refined_positions, polygon_select1.ind,axis=0)
#print(np.shape(remaining_peaks))
#fig, ax = plt.subplots()
#pts = ax.scatter(remaining_peaks[:,1], remaining_peaks[:,0])
#polygon_select = SelectFromCollection(ax, pts)
#plt.axis('square')
#plt.show()
#polygon_select1.disconnect()
# Computing Delaunay
#tri = Delaunay(np.array(platinum))

# Separating small and large edges:
#thresh = 45.0  # user defined threshold
#small_edges = set()
#large_edges = set()
#print(np.shape(tri.simplices))
#for tr in tri.simplices:
#    for i in [0,1,2]:
#        edge_idx0 = tr[i]
#        edge_idx1 = tr[(i+1)%3]
#        if (edge_idx1, edge_idx0) in small_edges:
#            continue  # already visited this edge from other side
#        if (edge_idx1, edge_idx0) in large_edges:
#            continue
#        p0 = platinum[edge_idx0]
#        p1 = platinum[edge_idx1]
#        if np.linalg.norm(p1 - p0) <  thresh:
#            small_edges.add((edge_idx0, edge_idx1))
#        else:
            #print(tr,"  ", tri.simplices)
             #print(tri.simplices)
            #tri.simplices = np.delete(tri.simplices,tr_index,axis=0)
            #print(np.shape(tri.simplices))
#            large_edges.add((edge_idx0, edge_idx1))

#print(np.shape(tri.simplices))
# Plotting the output
#plt.triplot(platinum[:,1], platinum[:,0], tri.simplices)
#plt.plot(platinum[:,1], platinum[:,0], 'o')
#plt.show()
#print(platinum)
#print(small_edges)
#plt.plot(platinum[:, 1], platinum[:, 0], '.')
#for i, j in small_edges:
#    plt.plot(platinum[[i, j], 0], platinum[[i, j], 1], 'b')
#for i, j in large_edges:
    #plt.plot(platinum[[i, j], 0], platinum[[i, j], 1], 'c')
#plt.show()

#A = dict(vertices=platinum)
#B = trrr.triangulate(A,['QJ'])
#trrr.compare(plt, A, B)

#plt.show()