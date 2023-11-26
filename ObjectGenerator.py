#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
import random
from mayavi import mlab
from perlin_noise import PerlinNoise
from scipy.ndimage import binary_dilation as dilate
from skimage.measure import label, regionprops
# %%
class ObjectGenerator():
    def __init__(self):
        self.support_lattice_constant = 5.3
        self.particle_lattice_constant = 2.8
        self.particle_radius = 5
        self.interface_spacing = 0.5
        self.epsilon = 1e-10

    def set_support_lattice_constant(self, support_lattice_constant):
        self.support_lattice_constant = support_lattice_constant

    def set_particle_lattice_constant(self, particle_lattice_constant):
        self.particle_lattice_constant = particle_lattice_constant

    def set_particle_radius(self, particle_radius):
        self.particle_radius = particle_radius

    def convex_hull(self, points):
        hull = ConvexHull(points)
        return hull

    def particle_hull(self,particle_layers,interface_radii=None,layer_sample_points=None, centers=None):
        if interface_radii is None:
            interface_radii = [self.particle_lattice_constant*particle_layers for i in range(particle_layers)]
        if layer_sample_points is None:
            layer_sample_points = [6 for i in range(particle_layers)]
        if centers is None:
            centers = [[0,0] for i in range(particle_layers)]

        layer_points = []
        for layer in range(particle_layers):
            points = []
            angles = []
            radius = interface_radii[layer]
            num_sample_points = layer_sample_points[layer]
            angle_offset = random.uniform(0,2*np.pi/num_sample_points)
            center = centers[layer]
            # get angles to sample points
            for i in range(num_sample_points):
                angles.append(random.uniform(i*2*np.pi/num_sample_points+self.epsilon,(i+1)*2*np.pi/num_sample_points-self.epsilon)+angle_offset)
            # sample points from circle with radius interface_radius
            for angle in angles:
                points.append([radius*np.cos(angle)+center[0],radius*np.sin(angle)+center[1],layer*self.particle_lattice_constant])
            
            layer_points.append(points)
        hull_sections = []
        for i in range(particle_layers):
            if i == particle_layers-1:
                points = layer_points[i] + layer_points[0]
            else:
                points = layer_points[i] + layer_points[i+1]
            hull_sections.append(self.convex_hull(points))
        return hull_sections

    def visualize_hull_points(self, hull_list):
        # Concatenate all points from each hull
        points = np.concatenate([hull.points for hull in hull_list])

        # Create a 1x3 grid of subplots
        fig, axes = plt.subplots(1, 3,figsize=(15,5), subplot_kw={'projection': '3d'})

        # Plot for X-axis view
        axes[0].scatter(points[:, 0], points[:, 1], points[:, 2])
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_zlabel('Z')
        axes[0].set_xticks([])
        axes[0].view_init(0, 0)
        # Plot for Y-axis view
        axes[1].scatter(points[:, 0], points[:, 1], points[:, 2])
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_zlabel('Z')
        axes[1].set_yticks([])
        axes[1].view_init(0, 90)
        # Plot for Z-axis view
        axes[2].scatter(points[:, 0], points[:, 1], points[:, 2])
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].set_zlabel('Z')
        axes[2].set_zticks([])
        axes[2].view_init(90, 0)
        # Adjust the aspect ratio
        for ax in axes:
            ax.set_box_aspect([1,1,1])  # Set equal aspect ratio

        plt.show()

    def mayavi_points(self, hull_list):
        # Concatenate all points from each hull
        points = np.concatenate([hull.points for hull in hull_list])

        # Modify these values as needed
        point_size = 5  # Adjust the size of the points
        point_color = (0,0.2, 1)  # Red color, as an example

        mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                    scale_factor=point_size, 
                    color=point_color, 
                    mode='sphere')  # Use 'sphere' for shaded points
        mlab.show()

    def expand_region_with_perlin(self, binary_map,structuring_element=np.array([[0,1,1,1,0],
                                                                                [1,1,1,1,1],
                                                                                [1,1,1,1,1],
                                                                                [1,1,1,1,1],
                                                                                [0,1,1,1,0]])):
        points = np.column_stack(np.where(binary_map == 1))
        noise = PerlinNoise(octaves=4, seed=random.randint(0,10000))
        xpix, ypix = binary_map.shape[0],binary_map.shape[1]
        pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
        pic = np.array(pic)
        noise_map = np.where(pic>0.1,1,0)
        #take the union of particle_map and noise_map
        new_map = np.where(noise_map+binary_map>0,1,0)
        new_map_labeled = label(new_map)
        new_map_props = regionprops(new_map_labeled)
        points_set = {tuple(point) for point in points}

        for region in new_map_props:
            region_coords_set = set(map(tuple, region.coords))
            if points_set.intersection(region_coords_set):
                new_map = np.where(new_map_labeled == region.label, 1, 0)
                break
        #new_map = dilate(new_map, structure=structuring_element)    
        return new_map

    def generate_particle_interface_map(self, particle_interface_points, width, depth, structuring_element=np.array([[0,1,1,1,0],
                                                                                                                    [1,1,1,1,1],
                                                                                                                    [1,1,1,1,1],
                                                                                                                    [1,1,1,1,1],
                                                                                                                    [0,1,1,1,0]])):
        particle_interface_points += [[width/2,depth/2]]
        hull = self.convex_hull(particle_interface_points)
        delaunay_hull = Delaunay(particle_interface_points[hull.vertices])        
        particle_map = np.zeros((width, depth))
        for i in range(width):
            for j in range(depth):
                if delaunay_hull.find_simplex([i, j]) >= 0:
                    particle_map[i, j] = 1
        particle_map = dilate(particle_map, structure=structuring_element)
        particle_map = dilate(particle_map, structure=structuring_element)
        particle_map = dilate(particle_map, structure=structuring_element)
        return particle_map

    def get_positions_from_binary_map(self, binary_map):
        positions = np.column_stack(np.where(binary_map == 1))
        valid_positions = []

        rows, cols = binary_map.shape

        for pos in positions:
            x, y = pos
            # Check if the position is on the edge
            if x == 0 or x == rows - 1 or y == 0 or y == cols - 1:
                valid_positions.append((x, y))
                continue

            # Check adjacent positions
            adjacent = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            if any(binary_map[adj_x, adj_y] == 0 for adj_x, adj_y in adjacent):
                valid_positions.append((x, y))
        #return the points of the convex hull of valid positions
        hull = ConvexHull(valid_positions)
        hull_points = hull.points[hull.vertices]
        return np.array(hull_points)

    def add_z_coordinate(self, points, z):
        points = np.column_stack((points,np.ones(points.shape[0])*z))
        return points

    def support_hull(self, support_layers, depth, width,particle_interface_points=None, structuring_element=None):
        
        #extract x and y coordinates from particle_interface_points
        if particle_interface_points is not None:
            xy_interface_points = particle_interface_points[:,0:2]
            if structuring_element is None:
                particle_map = self.generate_particle_interface_map(xy_interface_points, width, depth)
            else:
                particle_map = self.generate_particle_interface_map(xy_interface_points, width, depth, structuring_element)
        else :
            particle_map = np.zeros((width,depth))
        
        plt.figure()
        plt.imshow(particle_map, cmap='gray')
        print("particle_map")
        plt.show()

        surface = self.expand_region_with_perlin(particle_map)

        surface_points = self.get_positions_from_binary_map(surface)-np.array([width/2,depth/2])
        surface_points = self.add_z_coordinate(surface_points,0)


        hull_sections = []
        for layer in range(support_layers):
            new_map = self.expand_region_with_perlin(surface)
            new_positions = self.get_positions_from_binary_map(new_map)-np.array([width/2,depth/2])
            new_positions = self.add_z_coordinate(new_positions,(layer+1)*self.support_lattice_constant*-1)
            hull_section_points = np.concatenate((surface_points,new_positions))
            hull_sections.append(self.convex_hull(hull_section_points))
            surface = new_map
            surface_points = new_positions
        return hull_sections

    def generate_atomic_structure(self):

        atoms = {}
        return atoms

# %%
ob_gen = ObjectGenerator()
particle_hull = ob_gen.particle_hull(4,interface_radii=[6,7,6,5],layer_sample_points=[6,6,6,6],centers=[[0,1],[0,0],[0,0],[0,0]])
print(particle_hull[0].points)
# %%
support_hull = ob_gen.support_hull(8,100,100,particle_interface_points=np.array([[5.69638606, 2.88445904, 0],[0.30238439, 6.99237546, 0],[-4.64750397, 4.79482633, 0],[-5.62285321, -1.09368618, 0],[-1.00215511, -4.9157151, 0],[5.76586847, -0.65974722, 0]]))
print(len(particle_hull),len(support_hull))
ob_gen.visualize_hull_points(support_hull)
ob_gen.mayavi_points(support_hull+particle_hull)
# %%
bob = np.array([[0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]])

# Create a 5x5 structuring element
structuring_element = np.array([[0,1,1,1,0],
                                [1,1,1,1,1],
                                [1,1,1,1,1],
                                [1,1,1,1,1],
                                [0,1,1,1,0]])

# Apply dilation
bob_dilated = dilate(bob, structure=structuring_element)
print(bob_dilated)
particle_points = np.column_stack(np.where(bob == 1))
print(particle_points)
# %%
