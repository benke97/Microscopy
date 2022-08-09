import stemtool as st
import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io
from Polygon_selector import SelectFromCollection

def get_column_positions(im, thresh2, dist2,pixel_size):
    #im = skimage.io.imread("test.tif")
    atoms = st.afit.atom_fit(im,pixel_size, 'nm')
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

if __name__ == "__main__":
    im = skimage.io.imread("test2.tif")
    refined_positions = get_column_positions(im,0.1,0.1,0.009)
    x = refined_positions[:,1]
    y = refined_positions[:,0]

    im = plt.imread("test2.tif")
    implot = plt.imshow(im)

    plt.scatter(y, x, c='r', s=40)
    plt.show()

    fig, ax = plt.subplots()
    pts = ax.scatter(x, y)
    polygon_select1 = SelectFromCollection(ax, pts)
    plt.axis('square')
    plt.show()
    polygon_select1.disconnect()
    platinum = polygon_select1.xys[polygon_select1.ind]
    names = 'boitest'
    phases = platinum

    fig, ax = plt.subplots()
    coll = ax.scatter(phases[:,0], phases[:,1], color=["blue"]*len(phases), picker = 5, s=[50]*len(phases))
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
        print(phases[event.ind], "clicked")

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
    from write_PSLG import write_PSLG
    write_PSLG(names,phases,edge_loop_idx)

    import triangle as tr
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    particle = tr.load(dir_path, str(names))
    t = tr.triangulate(particle, 'pne')

    tr.compare(plt, particle, t)
    plt.show()
    from material import Material
    a = Material(t)

    boi = ax.tripcolor(np.array(a.vertices)[:,0],np.array(a.vertices)[:,1], np.array(a.triangles),facecolors=np.array(a.trig_strain), cmap='coolwarm',alpha=0.7, edgecolors='k')
    fig.colorbar(boi)
    #print(np.amax(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
    #print(np.amin(np.append(np.array(platin.trig_strain),np.array(ceri.trig_strain))))
    plt.show()
