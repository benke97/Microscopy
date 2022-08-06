import stemtool as st
import matplotlib.pyplot as plt
import skimage
import numpy as np

def get_column_positions(im, thresh2, dist2):
    #im = skimage.io.imread("test.tif")
    atoms = st.afit.atom_fit(im,0.009, 'nm')
    #image_max = ndi.maximum_filter(im, size=5, mode='constant')
    #coordinates = peak_local_max(im, min_distance=10)
    #threshold = 15

    #for coordinate in coordinates:
    #    if im[coordinate[0],coordinate[1]] < threshold:
    #        coordinates = np.delete(coordinates, np.where(np.all(coordinates==coordinate,axis=1)), axis=0)
    atoms.peaks_vis(dist=dist2, thresh=thresh2)
    atoms.refine_peaks()
    plt.close()
    x = atoms.refined_peaks[:,1]
    y = atoms.refined_peaks[:,0]
    return np.column_stack((x,y))