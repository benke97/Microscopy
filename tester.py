import skimage
from skimage import io
import triangle as tr
import os 
from material import Material
from Plotter import Plotter


"Enter image and the names of the phases"
#5 LAYER PARTICLE
#im = skimage.io.imread("test.tif")
#names = ['platinum','cerium']

#LARGE SUPPORT
im = skimage.io.imread("1601_110822.tif")
names = ['boitest','boitest_particle']

#4 LAYER PARTICLE
#im = skimage.io.imread("1601_110822_2.tif")
#names = ['1601zoomsupport','1601zoomparticle']

#SQUARE GRID
#im = skimage.io.imread("test3.tif")
#names = ['squaregrid']

materials = []
for n in range(len(names)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names[n]))
    t = tr.triangulate(particle, 'pne')
    materials.append(Material(t))


a = Plotter(materials,names,im)
#a.displacement_fields(names)
a.plot_displacement_field(names)
a.plot_strain_tensor(names)
a.plot_voronoi(names,strain=True)
a.plot_displacement_map(names)
a.plot_delaunay(names,strain=True)
print(im.shape)