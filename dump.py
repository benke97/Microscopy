    def calc_ideal_voronoi_areas(self,vertex,sorted_trig_idx,edge): #KOMPLETT BS
        new_values = self.voronoi_verts
        nbors = self.connections[vertex]
        a = np.array(self.vertices)[np.array(nbors)] - np.array(self.vertices[vertex])
        sort_idx = np.argsort(np.arctan2(a[:,0],a[:,1])*360/pi)
        sorted_neighbors = np.array(nbors)[sort_idx]
        print(len(sorted_trig_idx),len(self.connections[vertex]))
        if edge:
            print('a')
        else:
            points1 = []
            points2 = []
            for trig in np.array(self.triangles)[sorted_trig_idx]:
                a,b = self.circumcenter_ideal(vertex,trig)
                c,d = self.circumcenter(trig)
                points1.append([a,b])
                points2.append([c,d])
                print('b', trig)
            [self.points.append(point_pair) for point_pair in points1]
            [self.points2.append(point_pair) for point_pair in points2]
            print(points1,points2)
            voronoi_area1 = self.cell_area(np.array(points1))
            voronoi_area2 = self.cell_area(np.array(points2))
        #voronoi_verts_array = np.zeros((len(self.ideal_vertices),max(len(con_sl) for con_sl in self.connections)))
            print(voronoi_area1,voronoi_area2)

    def circumcenter_ideal(self,vertex,trig): #KOMPLETT BULLSHIT
        if trig[0] == vertex:
            ax = self.ideal_vertices[trig[0]][0]
            ay = self.ideal_vertices[trig[0]][1]
        else:
            ax = self.vertices[trig[0]][0]
            ay = self.vertices[trig[0]][1]   
        if trig[1] == vertex:         
            bx = self.ideal_vertices[trig[1]][0]
            by = self.ideal_vertices[trig[1]][1]
        else:
            bx = self.vertices[trig[1]][0]
            by = self.vertices[trig[1]][1]

        if trig[2] == vertex:    
            cx = self.ideal_vertices[trig[2]][0]
            cy = self.ideal_vertices[trig[2]][1]
        else:
            cx = self.vertices[trig[2]][0]
            cy = self.vertices[trig[2]][1]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return (ux, uy)

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