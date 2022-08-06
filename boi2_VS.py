import stemtool as st
import cv2
import skimage
import numpy as np
from skimage import io
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
im = skimage.io.imread("test.tif")
atoms = st.afit.atom_fit(im,0.009, 'nm')
#atoms.show_image()
#img = img_as_float(data.coins())
image_max = ndi.maximum_filter(im, size=5, mode='constant')
coordinates = peak_local_max(im, min_distance=10)
threshold = 15

for coordinate in coordinates:
    if im[coordinate[0],coordinate[1]] < threshold:
        coordinates = np.delete(coordinates, np.where(np.all(coordinates==coordinate,axis=1)), axis=0)
#atoms.define_reference((150,400), (400,400), (400,150), (150,150))
atoms.peaks_vis(dist=0.2, thresh=0.1)
atoms.refine_peaks()
atoms.show_peaks(style = 'together')
x = atoms.refined_peaks[:,1]
y = atoms.refined_peaks[:,0]
Fitted_coordinates = np.column_stack((x,y))
#print(Fitted_coordinates)
#plt.clf()
#plt.ginput(4)
#fig = plt.plot(Fitted_coordinates[:, 1], Fitted_coordinates[:, 0], 'k.')
#plt.show()


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

"""
Compute the mean and stddev of 100 data sets and plot mean vs. stddev.
When you click on one of the (mean, stddev) points, plot the raw dataset
that generated that point.
"""

fig, ax = plt.subplots()
ax.set_title('click on points')

line, = ax.plot(x,y, 'o',
                picker=True, pickradius=5)  # 5 points tolerance

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)

fig.canvas.mpl_connect('pick_event', onpick)
plt.axis('square')
plt.show()