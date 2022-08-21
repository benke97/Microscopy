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
                trig_strain = np.array(element.trig_rel_size)
            else:
                vertices = np.concatenate((vertices,np.array(element.vertices)),axis=0)
                triangles = np.concatenate((triangles,np.array(element.triangles)+np.amax(triangles+1)),axis=0)
                trig_strain = np.append(trig_strain,np.array(element.trig_rel_size))
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
        p.set_clim([-np.amax(np.abs(trig_strain)),np.amax(np.abs(trig_strain))])
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
        p = PatchCollection(patches,alpha = 0.5,cmap='coolwarm')
        p.set_array(colors,)
        p.set_clim([-np.amax(np.abs(colors)),np.amax(np.abs(colors))])
        #p.set_clim([-15,15])
        ax.add_collection(p)
        fig.colorbar(p)
        plt.show()

    def plot_strain_tensor(self,names):
        fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(nrows=2,ncols=2)
        im_data = self.im.T
        ax1.imshow(im_data,origin = 'lower',cmap = 'gray')
        ax2.imshow(im_data,origin = 'lower',cmap = 'gray')
        ax3.imshow(im_data,origin = 'lower',cmap = 'gray')
        ax4.imshow(im_data,origin = 'lower',cmap = 'gray')
        ax1.title.set_text(r'$\varepsilon_{xx}$')
        ax2.title.set_text(r'$\varepsilon_{xy}$')
        ax3.title.set_text(r'$\varepsilon_{yy}$')
        ax4.title.set_text(r'$\Omega_{xy}$')
        
        vertices = []
        triangles = []
        trig_strain = []
        for name in names:
            element = self.get_material(name)
            if vertices == []:
                vertices = np.array(element.vertices)
                triangles = np.array(element.triangles)
                eps_xx = np.array(element.trig_norm_strain_x)*100
                eps_xy = np.array(element.trig_shear_strain)*100
                eps_yy = np.array(element.trig_norm_strain_y)*100
                omega_xy = np.array(element.trig_rotation)*100
            else:
                vertices = np.concatenate((vertices,np.array(element.vertices)),axis=0)
                triangles = np.concatenate((triangles,np.array(element.triangles)+np.amax(triangles+1)),axis=0)
                eps_xx = np.append(eps_xx,np.array(element.trig_norm_strain_x)*100)
                eps_xy = np.append(eps_xy,np.array(element.trig_shear_strain)*100)
                eps_yy = np.append(eps_yy,np.array(element.trig_norm_strain_y)*100)
                omega_xy = np.append(omega_xy,np.array(element.trig_rotation)*100)

        for name in names:
            element = self.get_material(name)
            a = [np.vstack((np.array(element.vertices)[seg])) for seg in element.edges]
            line_segments1 = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid',alpha=0.5)
            line_segments2 = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid',alpha=0.5) 
            line_segments3 = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid',alpha=0.5)
            line_segments4 = LineCollection(a, linewidths=2,
                                colors='k', linestyle='solid',alpha=0.5)                                            
            ax1.add_collection(line_segments1)  
            ax2.add_collection(line_segments2)
            ax3.add_collection(line_segments3)
            ax4.add_collection(line_segments4)   
        p1 = ax1.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=eps_xx, cmap='coolwarm',alpha=0.7, edgecolors='k')
        p2 = ax2.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=eps_xy, cmap='coolwarm',alpha=0.7, edgecolors='k')
        p3 = ax3.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=eps_yy, cmap='coolwarm',alpha=0.7, edgecolors='k')
        p4 = ax4.tripcolor(vertices[:,0],vertices[:,1],triangles,facecolors=omega_xy, cmap='coolwarm',alpha=0.7, edgecolors='k')
        p1.set_clim([-np.amax(np.abs(eps_xx)),np.amax(np.abs(eps_xx))])
        p2.set_clim([-np.amax(np.abs(eps_xy)),np.amax(np.abs(eps_xy))])
        p3.set_clim([-np.amax(np.abs(eps_yy)),np.amax(np.abs(eps_yy))])
        p4.set_clim([-np.amax(np.abs(omega_xy)),np.amax(np.abs(omega_xy))])
        fig.colorbar(p1,ax=ax1)
        fig.colorbar(p2,ax=ax2)
        fig.colorbar(p3,ax=ax3)
        fig.colorbar(p4,ax=ax4)
        plt.show()