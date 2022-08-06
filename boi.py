import stemtool as st
import skimage
from skimage import io
from IPython.display import Image
Image(filename='peaks.jpg') 
image = skimage.io.imread("peaks.jpg")
atoms = st.afit.atom_fit(image,0.009, 'nm')
atoms.show_image(12)