import matplotlib.pyplot as plt
import triangle as tr
import numpy as np
import os 
import skimage
from skimage import io
from material import Material


names = 'boitest'
dir_path = os.path.dirname(os.path.realpath(__file__))
particle = tr.load(dir_path, str(names))
t = tr.triangulate(particle, 'pne')

tr.compare(plt, particle, t)
plt.show()
a = Material(t)

im = skimage.io.imread("test2.tif")
fig, ax = plt.subplots()
im_data = im.T
imaa = ax.imshow(im_data,origin = 'lower',cmap = 'gray')
tr.plot(ax,**t)

boi = ax.tripcolor(np.array(a.vertices)[:,0],np.array(a.vertices)[:,1], np.array(a.triangles),facecolors=np.array(a.trig_strain), cmap='coolwarm',alpha=0.7, edgecolors='k')
fig.colorbar(boi)
#print(np.amax(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
#print(np.amin(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
plt.show()