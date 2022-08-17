import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

class Plotter:
    def __init__(self,materials,names,im):
        if len(materials) != len(names):
            raise Exception (f'materials and names must have the same length')
        self.materials = materials
        self.im = im
        self.names = names

    def get_material(self,name):
        return self.materials[self.names.index(name)]    
    
    def plot_displacement_map(self,names,**kwargs):
        cmap = 'gray'
        origin = 'lower'
        im_data = self.im.T
        fig, ax = plt.subplots()
        ax.imshow(im_data,origin=origin,cmap = cmap)
        for name in names:
            element = self.get_material(name)
            V = np.array(element.vertex_displacements[0])
            x,y = -V.T #IMPORTANT MINUS SIGN
            x_dir = x.tolist()
            y_dir = y.tolist()
            x,y = np.array(element.vertices).T
            x_pos = x.tolist()
            y_pos = y.tolist()
            ax.quiver(x_pos,y_pos,x_dir,y_dir,angles='xy', scale_units='xy', scale=1, color = 'r')
        plt.show()

    def plot_delaunay(self,names,strain=False,**kwargs):
        fig, ax = plt.subplots()
        im_data = self.im.T
        ax.imshow(im_data,origin = 'lower',cmap = 'gray')
        
        vertices = []
        triangles = []
        trig_strain = []
        for name in names:
            element = self.get_material(name)
            if vertices == []:
                vertices = np.array(element.vertices)
                triangles = np.array(element.triangles)
                trig_strain = np.array(element.trig_strain)
            else:
                vertices = np.concatenate((vertices,np.array(element.vertices)),axis=0)
                triangles = np.concatenate((triangles,np.array(element.triangles)+np.amax(triangles+1)),axis=0)
                trig_strain = np.append(trig_strain,np.array(element.trig_strain))
        for name in names:
            element = self.get_material(name)
            a = [np.vstack((np.array(element.vertices)[seg])) for seg in element.edges]
            line_segments = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid')
            ax.add_collection(line_segments)
        if strain:        
            p = ax.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=trig_strain, cmap='coolwarm',alpha=0.5, edgecolors='k')
            fig.colorbar(p)
        else:
            p = ax.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=trig_strain, cmap='coolwarm',alpha=0, edgecolors='k')
        p.set_clim([-35,35])
        #lägg till segments och verts

        #boi = ax.tripcolor(np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,0],np.concatenate((np.array(platin.vertices),np.array(ceri.vertices)),axis=0)[:,1],
        #np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0),facecolors=np.append(np.array(platin.trig_strain),
        #np.array(ceri.trig_strain)), cmap='coolwarm',alpha=0.5, edgecolors='k')
        #np.concatenate((np.array(platin.triangles),np.array(ceri.triangles)+np.amax(np.array(platin.triangles)+1)),axis=0)
        plt.show()
        #plot delaunay, optional arguments: include strain, plotstuff
        #print('thelawny')
        
        
    def plot_voronoi(self,names,strain=False,edges=False,**kwargs):
        #print(strain)
        im_data = self.im.T
        fig, ax = plt.subplots()
        ax.imshow(im_data,origin = 'lower',cmap = 'gray')
        patches = []
        colors = []
        for name in names:
            element = self.get_material(name)
            #print(element.edge_classes)
            #print(element.number_of_connections)
            if colors == []:
                colors = np.array(element.voronoi_rel_size)[element.voronoi_bulk]
            else:
                colors = np.append(colors,np.array(element.voronoi_rel_size)[element.voronoi_bulk])

            for cell_nr in range(len(element.voronoi_bulk)):
                polygon = Polygon(np.array(element.voronoi_verts)[np.array(element.voronoi_cells)[element.voronoi_bulk][cell_nr]], True)
                patches.append(polygon)
            a = [np.vstack((np.array(element.voronoi_verts)[seg])) for seg in element.voronoi_segs]
            line_segments = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid')
            ax.add_collection(line_segments)
        p = PatchCollection(patches,alpha = 0.7,cmap='coolwarm')
        p.set_array(colors,)
        p.set_clim([-15,15])
        ax.add_collection(p)
        fig.colorbar(p)
        plt.show()