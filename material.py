from sqlite3 import connect
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math 
from scipy.spatial import distance
from cmath import inf, pi
from collections import deque
from collections import Counter

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
        self.voronoi_cells = []
        self.voronoi_vertices = []
        self.voronoi_verts = []
        self.voronoi_segs = []
        self.central_vertex = 0
        self.primitive_vectors = np.zeros((2,2))
        self.neighbors = triangle_object['neighbors'].tolist()
        self.vertices = triangle_object['vertices'].tolist() 
        self.triangles = triangle_object['triangles'].tolist()
        self.edges = triangle_object['edges'].tolist()
        self.segments = triangle_object['segments'].tolist()
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
        self.calc_voronoi_cells_bad()
        
        self.calc_voronoi_verts()
        self.calc_voronoi_segs()
        self.calc_voronoi_cells()

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
            if class1c == 99 or class2c == 99:
                c_length = self.get_length(trig[2],trig[0])
            else:
                print(class1a,class1b,class1c,class2a,class2b,class2c)
                #c_length = (np.linalg.norm(np.array(self.center_neighborhood_vectors[class1c]))+np.absolute(np.array(self.center_neighborhood_vectors[class2c])))/2
                c_length = (math.sqrt(pow(self.center_neighborhood_vectors[class1c][0],2)+pow(self.center_neighborhood_vectors[class1c][1],2))
                            + math.sqrt(pow(self.center_neighborhood_vectors[class2c][0],2)+pow(self.center_neighborhood_vectors[class2c][1],2)))/2


            ideal_area = self.calc_trig_area(a_length,b_length,c_length)
            strain_percent = (1-self.triangle_areas[i]/ideal_area)*100
            self.trig_strain.append(strain_percent)
            i +=1
    
    def calc_voronoi_cells_bad(self):
        for vertex in range(len(self.vertices)):

            #SORT NEIGHBORS
            nbors = self.connections[vertex]
            a = np.array(self.vertices)[np.array(nbors)] - np.array(self.vertices[vertex])
            sort_idx = np.argsort(np.arctan2(a[:,0],a[:,1])*360/pi)
            sorted_neighbors = np.array(nbors)[sort_idx]
            voro_verts = []
            lines = []
            #CALCULATE PERPENDICULAR LINE TO nbor-vertex at point vertex + (nbor-vertex)/2
            for nbor in sorted_neighbors:
                self.get_edge_index(vertex,nbor)
                k = (self.vertices[nbor][0]-self.vertices[vertex][0])/(self.vertices[nbor][1]-self.vertices[vertex][1])
                k2 = -1/k
                intercept_point = np.array(self.vertices[vertex])+(np.array(self.vertices[nbor])-np.array(self.vertices[vertex]))/2
                m = intercept_point[1]-k2*intercept_point[0]
                perpendicular_line = [k2,m]
                lines.append(perpendicular_line)
                #print(perpendicular_line)
                #print(self.vertices[nbor],self.vertices[vertex],intercept_point)
                #np.array(self.vertices[nbor])-np.array(self.vertices[vertex])
            for i in range(len(sorted_neighbors)):
                j = i+1
                if i == len(sorted_neighbors)-1:
                    j = 0
                A = np.array([[1,-lines[i][0]],[1,-lines[j][0]]])
                b = np.array([lines[i][1],lines[j][1]])
                y, x = np.matmul(np.linalg.inv(A),b).tolist()
                voronoi_vertex = [x,y]
                voro_verts.append(voronoi_vertex)
                self.voronoi_vertices.append(voronoi_vertex)
            #self.voronoi_cells.append(voro_verts) 
        #print(self.voronoi_cells)
                
    def circumcenter(self,trig):
        ax = self.vertices[trig[0]][0]
        ay = self.vertices[trig[0]][1]
        bx = self.vertices[trig[1]][0]
        by = self.vertices[trig[1]][1]
        cx = self.vertices[trig[2]][0]
        cy = self.vertices[trig[2]][1]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return (ux, uy)

    def calc_voronoi_verts(self):
        for trig in self.triangles:
            x,y = self.circumcenter(trig)
            self.voronoi_verts.append([x,y])


    def calc_voronoi_segs(self):
        for trig in range(len(self.triangles)):
            nbor_trigs = self.neighbors[trig]
            for nbor in nbor_trigs:
                if nbor == -1 or [nbor,trig] in self.voronoi_segs:
                    continue
                self.voronoi_segs.append([trig,nbor])

    def sorted_triangles(self,vertex):
        connected_triangles_idx = self.get_connected_triangles(vertex) 
        nbors = self.connections[vertex]
        a = np.array(self.vertices)[np.array(nbors)] - np.array(self.vertices[vertex])
        sort_idx = np.argsort(np.arctan2(a[:,0],a[:,1])*360/pi)
        sorted_neighbors = np.array(nbors)[sort_idx]
        
        if any(vertex in boundary for boundary in self.segments):
            #print('i am boundary',vertex)
            #print(sorted_neighbors)
            sorted_neighbors = self.set_start_to_edge(vertex,sorted_neighbors)
            #print(sorted_neighbors)
        #print('bob')
    
        #print(np.array(self.triangles)[connected_triangles_idx])
        #print(sorted_neighbors)
        sorted_trigs=[]
        for trig in range(len(connected_triangles_idx)):
            if len(connected_triangles_idx) == len(sorted_neighbors):
                if trig == len(connected_triangles_idx)-1:
                    sorted_trigs.append([vertex,sorted_neighbors[trig],sorted_neighbors[0]])
                    continue
                sorted_trigs.append([vertex,sorted_neighbors[trig],sorted_neighbors[trig+1]])
            else:
                if trig == len(connected_triangles_idx):
                    sorted_trigs.append([vertex,sorted_neighbors[trig],sorted_neighbors[0]])
                    continue
                sorted_trigs.append([vertex,sorted_neighbors[trig],sorted_neighbors[trig+1]])

        i=0
        #print(sorted_trigs)
        for trig in sorted_trigs:

            for trigg in np.array(self.triangles)[connected_triangles_idx]:
                #print(self.compare(trig,trigg))
                if self.compare(trig,trigg):
                    sorted_trigs[i] = trigg.tolist()
            i+=1
   
        return sorted_trigs


    def set_start_to_edge(self,vertex,sorted_nbors):
        #print('boi')
        sorted_nbors_dq = deque(sorted_nbors)
        while not self.is_boundary(vertex,sorted_nbors_dq[0]) or not self.is_boundary(vertex,sorted_nbors_dq[-1]):
            sorted_nbors_dq.rotate(1)
        return list(sorted_nbors_dq)

    def compare(self,s, t):
        return Counter(s) == Counter(t)

    def is_boundary(self,idx1,idx2):
        return [idx1,idx2] in self.segments or [idx2,idx1] in self.segments

    def find_boundary_in_trig(self,trig,vertex):
        if self.is_boundary(trig[0],trig[1]) and (trig[0] == vertex or trig[1] == vertex):
            if [trig[0],trig[1]] in self.segments:
                return [trig[0],trig[1]]
            return [trig[1],trig[0]]
        if self.is_boundary(trig[1],trig[2]) and (trig[1] == vertex or trig[2] == vertex):
            if [trig[1],trig[2]] in self.segments:
                return [trig[1],trig[2]]
            return [trig[2],trig[1]]  
        if self.is_boundary(trig[2],trig[0]) and (trig[2] == vertex or trig[0] == vertex):
            if [trig[2],trig[0]] in self.segments:
                return [trig[2],trig[0]]
            return [trig[0],trig[2]]                 

    def calc_voronoi_cells(self):
        check_array = np.zeros((len(self.vertices),len(self.vertices)),dtype=int)
        print(check_array)
        for vertex in range(len(self.vertices)):
            #connected_triangles_idx = self.get_connected_triangles(vertex)
            sorted_connected_triangles = self.sorted_triangles(vertex)
            #print('\n')
            #print(sorted_connected_triangles)
            #connected_triangles_idx = self.get_connected_triangles(vertex)
            #print(np.array(self.triangles)[connected_triangles_idx].tolist())
            sorted_trig_idx=[]

            for trig in sorted_connected_triangles:
                sorted_trig_idx.append(self.triangles.index(trig))

            if any(vertex in boundary for boundary in self.segments):
                print('edge_cell')
                #print(vertex)
                #print(np.array(self.triangles)[sorted_trig_idx])
                #print(self.voronoi_segs)
                #print(vertex,np.array(self.triangles)[sorted_trig_idx])
                if len(sorted_trig_idx) == 1:
                    nbors = self.connections[vertex]
                    a = np.array(self.vertices)[np.array(nbors)] - np.array(self.vertices[vertex])
                    sort_idx = np.argsort(np.arctan2(a[:,0],a[:,1])*360/pi)
                    sorted_neighbors = np.array(nbors)[sort_idx]
                    edge_segment_1 = [vertex,sorted_neighbors[1]]
                    edge_segment_2 = [vertex,sorted_neighbors[0]] 
                else:
                    edge_segment_1 = self.find_boundary_in_trig(np.array(self.triangles)[sorted_trig_idx][-1],vertex)
                    edge_segment_2 = self.find_boundary_in_trig(np.array(self.triangles)[sorted_trig_idx][0],vertex)
                if edge_segment_1 == edge_segment_2:
                    print( "din mamma")
                #print(edge_segment_1,edge_segment_2)
                if edge_segment_1[0] == vertex:
                    new_point_1 = np.array(self.vertices[vertex]) + (np.array(self.vertices[edge_segment_1[1]])-np.array(self.vertices[vertex]))/2
                    #print(self.vertices[vertex],new_point_1.tolist(),self.vertices[edge_segment_1[1]])
                else:
                    new_point_1 = np.array(self.vertices[vertex]) + (np.array(self.vertices[edge_segment_1[0]])-np.array(self.vertices[vertex]))/2
                    #print(self.vertices[vertex],new_point_1.tolist(),self.vertices[edge_segment_1[0]])
                
                if edge_segment_2[0] == vertex:
                    new_point_2 = np.array(self.vertices[vertex]) + (np.array(self.vertices[edge_segment_2[1]])-np.array(self.vertices[vertex]))/2
                    #print(self.vertices[vertex],new_point_2.tolist(),self.vertices[edge_segment_2[1]])
                else:
                    new_point_2 = np.array(self.vertices[vertex]) + (np.array(self.vertices[edge_segment_2[0]])-np.array(self.vertices[vertex]))/2
                    #print(self.vertices[vertex],new_point_2.tolist(),self.vertices[edge_segment_2[0]])


                #print(edge_1,edge_1 in self.segments)
                #if not new_point_1.tolist() in self.voronoi_verts:
                if not check_array[edge_segment_1[0],edge_segment_1[1]]:
                    self.voronoi_verts.append(new_point_1.tolist())
                    check_array[edge_segment_1[0],edge_segment_1[1]] = self.voronoi_verts.index(new_point_1.tolist())
                    check_array[edge_segment_1[1],edge_segment_1[0]] = self.voronoi_verts.index(new_point_1.tolist())
                #if not self.vertices[vertex] in self.voronoi_verts:    
                self.voronoi_verts.append(self.vertices[vertex])
                #if not new_point_2.tolist() in self.voronoi_verts:
                if not check_array[edge_segment_2[0],edge_segment_2[1]]:
                    self.voronoi_verts.append(new_point_2.tolist())
                    check_array[edge_segment_2[0],edge_segment_2[1]] = self.voronoi_verts.index(new_point_2.tolist())
                    check_array[edge_segment_2[1],edge_segment_2[0]] = self.voronoi_verts.index(new_point_2.tolist())
                print('SIZE',np.shape(np.array(self.voronoi_verts)))
                print(check_array)
                #print(self.voronoi_verts.index(new_point_1.tolist()),self.voronoi_verts.index(self.vertices[vertex]),self.voronoi_verts.index(new_point_2.tolist()))   
                sorted_trig_idx.append(check_array[edge_segment_1[0],edge_segment_1[1]])
                sorted_trig_idx.append(self.voronoi_verts.index(self.vertices[vertex]))
                sorted_trig_idx.append(check_array[edge_segment_2[0],edge_segment_2[1]])
                print(sorted_trig_idx)
                
                
                self.voronoi_segs.append([sorted_trig_idx[-4],sorted_trig_idx[-3]])
                self.voronoi_segs.append([sorted_trig_idx[-3],sorted_trig_idx[-2]])
                self.voronoi_segs.append([sorted_trig_idx[-2],sorted_trig_idx[-1]])
                self.voronoi_segs.append([sorted_trig_idx[-1],sorted_trig_idx[0]])
                #self.voronoi_segs.append([sorted_trig_idx[-1],check_array[edge_segment_1[0],edge_segment_1[1]]])
                #self.voronoi_segs.append([check_array[edge_segment_1[0],edge_segment_1[1]],self.voronoi_verts.index(self.vertices[vertex])])
                #self.voronoi_segs.append([self.voronoi_verts.index(self.vertices[vertex]),check_array[edge_segment_2[0],edge_segment_2[1]]])
                #self.voronoi_segs.append([check_array[edge_segment_2[0],edge_segment_2[1]],sorted_trig_idx[0]])
                
                #self.voronoi_segs.append([sorted_trig_idx[-1],check_array[edge_segment_1[0],edge_segment_1[1]]])
                #self.voronoi_segs.append([self.voronoi_verts.index(new_point_1.tolist()),self.voronoi_verts.index(self.vertices[vertex])])
                #self.voronoi_segs.append([self.voronoi_verts.index(self.vertices[vertex]),self.voronoi_verts.index(new_point_2.tolist())])
                #self.voronoi_segs.append([self.voronoi_verts.index(new_point_2.tolist()),sorted_trig_idx[0]])
                #if [self.voronoi_verts.index(new_point_1.tolist()),self.voronoi_verts.index(new_point_2.tolist())] in self.voronoi_segs or [self.voronoi_verts.index(new_point_2.tolist()),self.voronoi_verts.index(new_point_1.tolist())] in self.voronoi_segs:
                #    print('fan',vertex)

                #sorted_trig_idx.append(index(halvvägs mellan vertex och neighbor edge som finns i sista triangeln))
                #sorted_trig_idx.append(self.vertices[vertex].index i voroverts))
                #sorted_trig_idx.append(index(halvvägs mellan vertex och neighbor edge som finns i första triangeln)
            else:
                #for trig in sorted_connected_triangles:
                #    sorted_trig_idx.append(self.triangles.index(trig))
                #print('\n')
                #print(sorted_trig_idx, sorted_connected_triangles)
                #print(self.triangles)
                #print(len(self.triangles))
                #print(len(self.voronoi_verts))

                ordered_voronoi_cell = np.array(self.voronoi_verts)[sorted_trig_idx]
                voronoi_area = self.cell_area(ordered_voronoi_cell)
                print(voronoi_area)
        print(sorted_trig_idx)


            #voronoi_verts_idx = connected_triangles_idx 
            #voronoi_verts = np.array(self.voronoi_verts)[voronoi_verts_idx]
            #print(sorted_connected_triangles)
            
            #print(voronoi_verts_idx,voronoi_verts)



    def get_connected_triangles(self,vertex_idx):
        connected_triangles = [self.triangles.index(trig) for trig in self.triangles if vertex_idx in trig]
        #print(connected_triangles)
        return connected_triangles

    
    def cell_area(self,verts):
        x = verts[:,0]
        y = verts[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))