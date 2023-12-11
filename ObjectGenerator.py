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
import pandas as pd
from ase.cluster import wulff_construction
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from scipy.spatial import Delaunay
# %%
class ObjectGenerator():
    def __init__(self):
        self.support_lattice_constant = 5.3
        self.particle_lattice_constant = 1.96 #Actually lattice spacing
        self.particle_radius = 5
        self.interface_spacing = 0.5
        self.epsilon = 1e-10
        self.wulff_db = {"Pt": {"surfaces": [(1, 0, 0),(1,1,0), (1, 1, 1)], "esurf": [1.86, 1.68, 1.49], "lc": 3.9242, "crystal": "fcc"}}

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
        assert particle_layers == len(interface_radii) == len(layer_sample_points) == len(centers)
        if interface_radii is None:
            interface_radii = [self.particle_lattice_constant*particle_layers for i in range(particle_layers)]
        if layer_sample_points is None:
            layer_sample_points = [6 for i in range(particle_layers)]
        if centers is None:
            centers = [[0,0] for i in range(particle_layers)]

        layer_points = []
        for layer in range(particle_layers):
            print(layer)
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
        for i in range(particle_layers-1):
            print(layer_points)
            print(i)
            points = layer_points[i+1]+layer_points[i]
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

    def mayavi_atomic_structure(self, atoms):
        # Define color and scale maps
        color_map = {
            0: (0.94, 0.92, 0.84),  # Eggshell
            1: (1, 0, 0),            # Red
            2: (0.4, 0.6, 0.7)      # Metallic blue-green-gray
        }
        scale_map = {
            0: 2,  # Scale factor for Ce
            1: 1,   # Scale factor for O
            2: 2.2    # Scale factor for Pt
        }

        # Create a Mayavi figure with a white background
        mlab.figure(bgcolor=(1, 1, 1))

        # Group by label and plot each group
        for label, group in atoms.groupby('label'):
            xyz = group[['x', 'y', 'z']].values
            mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                          mode='sphere', 
                          color=color_map.get(label, (0, 0, 0)),
                          scale_factor=scale_map.get(label, 1))  # Use scale_map for scale_factor

        # Display the plot
        mlab.show()

    def expand_region_with_perlin(self, binary_map,structuring_element=np.array([[0,1,1,1,0],
                                                                                [1,1,1,1,1],
                                                                                [1,1,1,1,1],
                                                                                [1,1,1,1,1],
                                                                                [0,1,1,1,0]])):
        points = np.column_stack(np.where(binary_map == 1))
        noise = PerlinNoise(octaves=3, seed=random.randint(0,10000))
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
        new_map = dilate(new_map, structure=structuring_element)    
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
        number_of_dilations = random.randint(1,3)
        for i in range(number_of_dilations):
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
    
    def rotate_points_around_axis(self, points, axis, angle):
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])
        rotated_points = rotation_matrix.dot(points.T).T
        return rotated_points

    def Pt_lattice(self, depth, width, height, zone_axis=np.array([0,1,0])):
        lattice_constant = 3.9158
        cell_zone_axis = np.array([0,1,0])

        unit_cell = np.array([
            (0, 0, 0),
            (0, 0.5, 0.5),  
            (0.5, 0.5, 0),
            (0.5,0,0.5)  
        ]) * lattice_constant
        
        num_atoms = depth * width * height * len(unit_cell)

        # Preallocate NumPy arrays for coordinates and labels
        x_coords = np.zeros(num_atoms)
        y_coords = np.zeros(num_atoms)
        z_coords = np.zeros(num_atoms)
        labels = np.array(['Pt'] * num_atoms)

        # Fill the arrays
        index = 0
        for x in range(depth):
            for y in range(width):
                for z in range(height):
                    for atom in unit_cell:
                        x_coords[index] = atom[0] + x * lattice_constant
                        y_coords[index] = atom[1] + y * lattice_constant
                        z_coords[index] = atom[2] + z * lattice_constant
                        index += 1

        #find middle of lattice and shift to origin
        x_coords -= np.mean(x_coords)
        y_coords -= np.mean(y_coords)
        z_coords -= np.mean(z_coords)

        points = np.column_stack((x_coords,y_coords,z_coords))
        points = self.rotate_points_around_axis(points, [-1,-1,0], 0.9553166181245093)
        points = self.rotate_points_around_axis(points, [0,0,1], 0.9553166181245093)

        # Create DataFrame
        lattice = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], 'label': labels})

        return lattice

    def Ceria_lattice(self, depth, width, height):
        lattice_constant = 5.4097
        unit_cell = np.array([
            (0, 0, 0),
            (0, 0.5, 0.5),  
            (0.5, 0.5, 0),
            (0.5,0,0.5),
            (0.25,0.25,0.25),
            (0.75,0.25,0.25),
            (0.25,0.25,0.75),
            (0.75,0.25,0.75),
            (0.25,0.75,0.25),
            (0.75,0.75,0.25),
            (0.25,0.75,0.75),
            (0.75,0.75,0.75),
        ]) * lattice_constant

        labels_unit_cell = np.array(['Ce','Ce','Ce','Ce','O','O','O','O','O','O','O','O'])

        num_atoms = depth * width * height * len(unit_cell)

        #repeat labels_unit_cell depth*width*height times
        labels = np.tile(labels_unit_cell,depth*width*height)

        # Preallocate NumPy arrays for coordinates and labels
        x_coords = np.zeros(num_atoms)
        y_coords = np.zeros(num_atoms)
        z_coords = np.zeros(num_atoms)

        # Fill the arrays
        index = 0
        for x in range(depth):
            for y in range(width):
                for z in range(height):
                    for atom in unit_cell:
                        x_coords[index] = atom[0] + x * lattice_constant
                        y_coords[index] = atom[1] + y * lattice_constant
                        z_coords[index] = atom[2] + z * lattice_constant
                        index += 1

        #find middle of lattice and shift to origin
        x_coords -= np.mean(x_coords)
        y_coords -= np.mean(y_coords)
        z_coords -= np.mean(z_coords)

        points = np.column_stack((x_coords,y_coords,z_coords))
        points = self.rotate_points_around_axis(points, [-1,-1,0], 0.9553166181245093)
        points = self.rotate_points_around_axis(points, [0,0,1], 0.9553166181245093)


        # Create DataFrame
        lattice = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], 'label': labels})

        return lattice

    def point_in_convex_hull(self, point, hull):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= 0)
            for eq in hull.equations)

    def filter_atoms_by_hull_section(self, atoms, hull):
        # atoms is pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'label': labels})
        min_x, min_y, min_z = np.min(hull.points, axis=0)
        max_x, max_y, max_z = np.max(hull.points, axis=0)

        within_bbox = (
            (atoms['x'] >= min_x) & (atoms['x'] <= max_x) &
            (atoms['y'] >= min_y) & (atoms['y'] <= max_y) &
            (atoms['z'] >= min_z) & (atoms['z'] <= max_z)
        )
        
        atoms_within_bbox = atoms[within_bbox]
        
        print("in 2", len(atoms_within_bbox),atoms_within_bbox.x)
        
        within_hull = atoms_within_bbox.apply(
            lambda row: self.point_in_convex_hull(np.array(row[['x', 'y', 'z']]), hull),
            axis=1
        )

        # Return the atoms that are within both the bounding box and the convex hull
        return atoms_within_bbox[within_hull]

    def filter_atoms_by_hull(self, atoms, hull_list):
        all_filtered_atoms = []

        for hull in hull_list:
            filtered_atoms = self.filter_atoms_by_hull_section(atoms, hull)
            if len(filtered_atoms) > 0:
                all_filtered_atoms.append(filtered_atoms)

        # Combine all filtered atoms into a single DataFrame
        if all_filtered_atoms:
            combined_filtered_atoms = pd.concat(all_filtered_atoms).drop_duplicates()
        else:
            combined_filtered_atoms = pd.DataFrame(columns=atoms.columns)

        return combined_filtered_atoms

    def random_point_in_hull(self, hull):
        # Generate random point in bounding box
        min_x, min_y = np.min(hull.points, axis=0)
        max_x, max_y = np.max(hull.points, axis=0)
        point = np.array([random.uniform(min_x,max_x),random.uniform(min_y,max_y)])
        # Check if point is within hull
        while not self.point_in_convex_hull(point, hull):
            point = np.array([random.uniform(min_x,max_x),random.uniform(min_y,max_y)])
        return point

    def get_line_equation(self, p1, p2):
        # Calculate the coefficients A, B, and C of the line equation: Ax + By = C
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    def get_intersection_points(self, hull, line):
        
        def line_intersection(A1, B1, C1, A2, B2, C2):
            # Calculate the intersection of two lines given by A1x + B1y = C1 and A2x + B2y = C2
            determinant = A1 * B2 - A2 * B1
            if determinant == 0:
                return None  # Lines are parallel or coincident
            x = (C1 * B2 - C2 * B1) / determinant
            y = (A1 * C2 - A2 * C1) / determinant
            return np.array([x, y])

        def is_point_on_line_segment(p, line_start, line_end):
            # Check if a point is on the line segment defined by line_start and line_end
            return np.min([line_start[0], line_end[0]]) <= p[0] <= np.max([line_start[0], line_end[0]]) and \
                np.min([line_start[1], line_end[1]]) <= p[1] <= np.max([line_start[1], line_end[1]])
        
        intersections = []
        A1, B1, C1 = self.get_line_equation(line[0], line[1])

        for simplex in hull.simplices:
            p1, p2 = hull.points[simplex[0]], hull.points[simplex[1]]
            A2, B2, C2 = self.get_line_equation(p1, p2)

            intersection = line_intersection(A1, B1, C1, A2, B2, C2)
            if intersection is not None and is_point_on_line_segment(intersection, p1, p2):
                intersections.append(intersection)

        return intersections

    def split_hull_by_z(self, hull):
        #find unique values of z in hull points
        #return list of lists of points with the same z value
        z_values = np.unique(hull.points[:,2])
        hull_levels = []
        for z in z_values:
            hull_levels.append(hull.points[hull.points[:,2]==z])

        return z_values, hull_levels

    def filter_points(self,points, line_points, center_point):
        A, B, C = self.get_line_equation(line_points[0], line_points[1])
        
        def side_of_line(point):
            return A * point[0] + B * point[1] - C

        center_side = side_of_line(center_point)
        
        filtered_points = []
        for point in points:
            if side_of_line(point) * center_side >= 0:  # Same side or on the line
                filtered_points.append(point)
        
        return filtered_points
    

        """
        Check if two arrays are equal with a given tolerance.
        """
        if arr1.shape != arr2.shape:
            return False
        return np.allclose(arr1, arr2, atol=tol)
    
    def add_step(self, particle_hull, support_hull, position=None, height=2):

        if height > len(particle_hull):
            height = len(particle_hull)

        #CROP PARTICLE HULLS
        interface_hull = particle_hull[0]
        interface_points = np.where(interface_hull.points[:,2]==0)
        interface_points = interface_hull.points[interface_points]
        interface_center = np.mean(interface_points[:,0:2],axis=0)
        if position is None:
            step_position = self.random_point_in_hull(self.convex_hull(interface_points[:,0:2]))
        else:
            step_position = position
        #step_position = np.array([0.1,0.1])
        vector = step_position-interface_center
        orthogonal_vector = np.array([-vector[1],vector[0]])
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)
        endpoint_1 = step_position+orthogonal_vector*25
        endpoint_2 = step_position-orthogonal_vector*25
        line = np.array([endpoint_1,endpoint_2])

        intersections = []
        hulls_to_change = particle_hull[0:height]
        for i, hull in enumerate(hulls_to_change):
            new_hull = []
            #splita hull i layers
            z_values, hull_levels = self.split_hull_by_z(hull)
            #filtrera points
            new_points = []
            for z, level in zip(z_values, hull_levels):
                filtered_points = self.filter_points(level, line, interface_center)
                intersection_points = self.get_intersection_points(self.convex_hull(np.array(level)[:,:2]), line)
                #print("intersection_points",len(intersection_points))
                intersection_points = self.add_z_coordinate(np.array(intersection_points),z)
                intersections.append(intersection_points)

                if len(intersection_points) == 2:
                    level_points = np.concatenate((filtered_points,intersection_points))
                else:
                    level_points = level
                new_points.append(level_points)
            #plot new_hull and particle_hull[i]
            new_hull = self.convex_hull(np.concatenate(new_points))
            old_hull = particle_hull[i]
            particle_hull[i] = new_hull

        #GENERATE SUPPORT HULL
        #print("old_hull",old_hull.points)
        max_z = max(old_hull.points[:,2])
        foundation = support_hull[0].points
        foundation = foundation[foundation[:,2]==0]
        filter_foundation = np.array(self.filter_points(foundation, line, interface_center))
        filtered_out_points = []
        for point in foundation:
            if np.any(np.all(point == filter_foundation, axis=1)):
                continue
            else:
                filtered_out_points.append(point)
        filtered_out_points = np.array(filtered_out_points)
        intersection_points = self.get_intersection_points(self.convex_hull(foundation[:,:2]), line)
        intersection_points = self.add_z_coordinate(np.array(intersection_points),0)
        step_foundation = np.concatenate((filtered_out_points,intersection_points))
        
        old_hull_points_max_z = old_hull.points[old_hull.points[:,2]==max_z]
        print("old_hull_points and max_z",old_hull_points_max_z,max_z)
        intersection_points = self.get_intersection_points(self.convex_hull(np.array(old_hull_points_max_z)[:,:2]), line)
        if intersection_points == []:
            #step_foundation but with z = max_z
            step_top_points = step_foundation.copy()
            step_top_points[:,2] = max_z
            step_support_hull = self.convex_hull(np.concatenate((step_foundation,step_top_points)))
            support_hull.append(step_support_hull)
        else:
            filtered_old_hull = self.filter_points(old_hull_points_max_z, line, interface_center)
            filtered_out_old_hull = []
            for point in old_hull_points_max_z:
                if np.any(np.all(point == filtered_old_hull, axis=1)):
                    continue
                else:
                    filtered_out_old_hull.append(point)
            filtered_out_old_hull = np.array(filtered_out_old_hull)
            intersection_points = self.add_z_coordinate(np.array(intersection_points),max_z)

            step_old_hull = np.concatenate((filtered_out_old_hull,intersection_points))
            step_support_hull = self.convex_hull(np.concatenate((step_foundation,step_old_hull)))
            support_hull.append(step_support_hull)

        plt.figure()
        plt.scatter(foundation[:,0],foundation[:,1],c='red')
        plt.scatter(particle_hull[0].points[:,0],particle_hull[0].points[:,1],c='blue')
        plt.scatter(filter_foundation[:,0],filter_foundation[:,1],c='green')
        plt.scatter(interface_center[0],interface_center[1])
        plt.scatter(filtered_out_points[:,0],filtered_out_points[:,1],c='orange')
        #plt.scatter(step_old_hull[:,0],step_old_hull[:,1])
        plt.show()


        #find the points that were filtered out
        return particle_hull, support_hull 
    
    def generate_wulff_particle(self, size, element, rounding="above"):
        atoms = wulff_construction(element, 
                                      self.wulff_db[element]["surfaces"], 
                                      self.wulff_db[element]["esurf"], 
                                      size,self.wulff_db[element]["crystal"], 
                                      rounding=rounding, 
                                      latticeconstant=self.wulff_db[element]["lc"])
        points = atoms.positions
        points = self.rotate_points_around_axis(points, [-1,-1,0], 0.9553166181245093)
        points = self.rotate_points_around_axis(points, [0,0,1], 0.9553166181245093)
        particle = pd.DataFrame(points, columns=['x', 'y', 'z'])
        particle['label'] = 'Pt'

        # Set z to 0 if close to 0, and remove points with z < 0
        epsilon = 0.0001
        particle.loc[abs(particle['z']) < epsilon, 'z'] = 0
        particle = particle[particle['z'] >= 0]

        # Reset index and round z values
        particle.reset_index(drop=True, inplace=True)
        particle['z'] = particle['z'].round(4)
        self.mayavi_atomic_structure(particle)
        return particle

    def hull_from_points(self, points):
        "Assuming there are distinct levels of z values"
        z_values = np.unique(points[:,2])
        hull = []
        print(z_values)
        for i,z in enumerate(z_values):
            if i < len(z_values)-1:
                print(z,z_values[i+1])
                hull_section_points = np.concatenate((points[points[:,2]==z],points[points[:,2]==z_values[i+1]]))
                hull.append(self.convex_hull(hull_section_points))
        return hull
    
    def expand_hull(self, hull, val):
        new_hull = []
        for hull_section in hull:
            zs, levels = self.split_hull_by_z(hull_section)
            new_hull_section = []
            for level in levels:
                center_of_level = np.mean(level, axis=0)
                print("level",level)
                print("center_of_level",center_of_level)
                new_level = []
                for point in level:
                    # Create expansion vector
                    expansion_vector = point - center_of_level
                    expansion_vector /= np.linalg.norm(expansion_vector[:2])  # Normalize only x and y
                    expansion_vector[2] = 0  # Keep z-coordinate unchanged
                    new_point = point + expansion_vector * val
                    new_level.append(new_point)
                new_hull_section.append(new_level)
            new_hull.append(self.convex_hull(np.concatenate(new_hull_section)))

        return new_hull

    def set_interface_spacing(self, structure_df, interface_spacing):
        #find lowest z value of particle
        particle = structure_df[structure_df['label'] == 'Pt']
        lowest_z = np.min(particle['z'])
        #find highest z value of support lower than lowest z value of particle
        support = structure_df[structure_df['label'] == 'Ce']
        support = support[support['z'] < lowest_z]
        highest_z = np.max(support['z'])
        curr_interface_spacing = lowest_z - highest_z
        #shift particle up by interface_spacing - curr_interface_spacing, edit structure_df
        structure_df.loc[structure_df['label'] == 'Pt', 'z'] += interface_spacing - curr_interface_spacing
        return structure_df




    def relax_structure(self, structure_df):
        positions = structure_df[['x', 'y', 'z']].values
        labels = structure_df['label'].values
        new_positions = positions + np.random.normal(0, 0.1, size=positions.shape) 
        new_df = pd.DataFrame(new_positions, columns=['x', 'y', 'z'])
        new_df['label'] = labels
        return new_df

    def remove_overlapping_atoms(self, atom_df, support=["Ce", "O"], particle=["Pt"], threshold=1.5):
        """
        Remove atoms from the support that are too close to atoms in the particle.

        Parameters:
        atom_df (DataFrame): Pandas DataFrame with columns x, y, z, and label.
        support (list): List of labels for support atoms.
        particle (list): List of labels for particle atoms.
        threshold (float): Distance threshold for removing atoms (in angstroms).

        Returns:
        DataFrame: Modified DataFrame with overlapping atoms removed.
        """
        # Filter DataFrame for support and particle atoms
        support_atoms = atom_df[(atom_df['label'].isin(support)) & (atom_df['z'] > -5)]
        particle_atoms = atom_df[atom_df['label'].isin(particle)]

        # Identifying support atoms to remove
        atoms_to_remove = []
        for _, support_atom in support_atoms.iterrows():
            for _, particle_atom in particle_atoms.iterrows():
                # Calculate the distance between the support and particle atom
                distance = ((support_atom['x'] - particle_atom['x'])**2 +
                            (support_atom['y'] - particle_atom['y'])**2 +
                            (support_atom['z'] - particle_atom['z'])**2)**0.5
                if distance < threshold:
                    atoms_to_remove.append(support_atom.name)
                    break  # No need to check other particle atoms for this support atom

        # Remove identified atoms
        atom_df = atom_df.drop(atoms_to_remove)
        return atom_df
    
    def generate_atomic_structure(self):
        #generate bulk Pt and bulk CeO2
        Pt_bulk = self.Pt_lattice(10,10,10)
        CeO2_bulk = self.Ceria_lattice(10,10,10)
        #rotate to zone axis

        #generate particle and support hulls
        particle_hull = self.particle_hull(...)
        support_hull = self.support_hull(...)


        #Filter Pt_bulk by particle_hull and CeO2_bulk by support_hull
        filtered_Pt = self.filter_atoms_by_hull(Pt_bulk,particle_hull)
        filtered_ceo2 = self.filter_atoms_by_hull(CeO2_bulk,support_hull)


        return 0

# %%
ob_gen = ObjectGenerator()
#particle_hull = ob_gen.particle_hull(4,interface_radii=[10,11,8,7],layer_sample_points=[6,6,6,6],centers=[[0,0],[0,0],[0,0],[0,0]])
particle_hull = ob_gen.particle_hull(5,interface_radii=[10,11,9,8,7],layer_sample_points=[6,6,6,6,6],centers=[[0,0],[0,0],[0,0],[0,0],[0,0]])
#print(particle_hull[0].points)
#for i in range(100):
#    ob_gen.add_step(particle_hull.copy(),particle_hull.copy())
# %%
support_hull = ob_gen.support_hull(8,50,50,particle_interface_points=np.array([[5.69638606, 2.88445904, 0],[0.30238439, 6.99237546, 0],[-4.64750397, 4.79482633, 0],[-5.62285321, -1.09368618, 0],[-1.00215511, -4.9157151, 0],[5.76586847, -0.65974722, 0]]))
print(len(particle_hull),len(support_hull))
ob_gen.visualize_hull_points(support_hull+particle_hull)
ob_gen.mayavi_points(support_hull+particle_hull)
#stepped particle
particle_hull_2, support_hull_2 = ob_gen.add_step(particle_hull.copy(), support_hull) 
# %%
Pt_bulk = ob_gen.Pt_lattice(10,10,10)
#Pt_bulk.z -= 1
CeO2_bulk = ob_gen.Ceria_lattice(15,15,15)
#RANDOMLY DISPLACE BULK ATOMS BEFORE FILTERING


#plot the lattice, color by label
#translate label to color
filtered_Pt=ob_gen.filter_atoms_by_hull(Pt_bulk,particle_hull)
filtered_Pt_2=ob_gen.filter_atoms_by_hull(Pt_bulk,particle_hull_2)
filtered_ceo2 = ob_gen.filter_atoms_by_hull(CeO2_bulk,support_hull)
Pt_bulk.label = Pt_bulk.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
CeO2_bulk.label = CeO2_bulk.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(CeO2_bulk.x, CeO2_bulk.y, CeO2_bulk.z, c=CeO2_bulk.label)
ax.view_init(0, 45)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
#%%
#plot filtered pt and filtered ceo2 in the same frame with mayavi
filtered_atoms = pd.concat([filtered_Pt_2,filtered_ceo2])
filtered_atoms_2 = pd.concat([filtered_Pt,filtered_ceo2])
filtered_atoms_2.label = filtered_atoms_2.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
filtered_atoms.label = filtered_atoms.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms.x, filtered_atoms.y, filtered_atoms.z, c=filtered_atoms.label)
ax.view_init(0, 45)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms.x, filtered_atoms.y, filtered_atoms.z, c=filtered_atoms.label)
ax.view_init(90,0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms_2.x, filtered_atoms_2.y, filtered_atoms_2.z, c=filtered_atoms_2.label)
ax.view_init(90,0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
#plot particle_hull and particle_hull_2
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(particle_hull[0].points[:,0], particle_hull[0].points[:,1], particle_hull[0].points[:,2], color='red',s=100)
#ax.scatter(particle_hull[1].points[:,0], particle_hull[1].points[:,1], particle_hull[1].points[:,2], color='red')
#ax.scatter(particle_hull_2[0].points[:,0], particle_hull_2[0].points[:,1], particle_hull_2[0].points[:,2], color='blue')
#ax.scatter(particle_hull_2[1].points[:,0], particle_hull_2[1].points[:,1], particle_hull_2[1].points[:,2], color='blue')
#ax.view_init(0, 45)
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#ax.axis('equal')
#plt.show()


#print number of Pt atoms
print(len(filtered_Pt))
print(len(filtered_Pt_2))

#print z values of Pt atoms and unique z values of hull points



#find the points that are in filtered Pt but not in filtered Pt_2 and print their z coordinate
filtered_Pt_2_points = filtered_Pt_2[['x','y','z']].values
filtered_Pt_points = filtered_Pt[['x','y','z']].values
z_values = []
for point in filtered_Pt_points:
    if not np.any(np.all(point == filtered_Pt_2_points,axis=1)):
        z_values.append(point[2])
print(z_values)

ob_gen.mayavi_atomic_structure(filtered_atoms)


# %%# %%
from ase.cluster import wulff_construction

surfaces = [(1, 0, 0),(1,1,0), (1, 1, 1)]
esurf = [1.86, 1.68, 1.49]   # Surface energies.
lc = 3.9242
size = 200# Number of atoms
atoms = wulff_construction('Pt', surfaces, esurf,
                           size, 'fcc',
                           rounding='above', latticeconstant=lc)
#plot atoms with mayavi
#extract x,y,z coordinates from atoms
x_coords = atoms.positions[:,0]
y_coords = atoms.positions[:,1]
z_coords = atoms.positions[:,2]
particle = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'label': ['Pt']*len(x_coords)})
#rotate points around axis
points = np.column_stack((x_coords,y_coords,z_coords))
points = ob_gen.rotate_points_around_axis(points, [-1,-1,0], 0.9553166181245093)
points = ob_gen.rotate_points_around_axis(points, [0,0,1], 0.9553166181245093)
particle = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], 'label': ['Pt']*len(x_coords)})

#if z closer to 0 than epsilon, set z to 0, if z < -epsilon, remove point
epsilon = 0.0001

particle.loc[abs(particle['z']) < epsilon, 'z'] = 0
particle = particle[particle['z'] >= 0]

particle.reset_index(drop=True, inplace=True)
particle['z'] = particle['z'].round(4)
ob_gen.mayavi_atomic_structure(particle)

print(len(particle))
#print number of atoms
#print unique z values
print(np.unique(particle.z))
#print number of atoms at each z
for z in np.unique(atoms.positions[:,2]):
    print(len(atoms.positions[atoms.positions[:,2]==z]))

# %%
ob_gen = ObjectGenerator()
bob = ob_gen.generate_wulff_particle(200,'Pt')
CeO2_bulk_test = ob_gen.Ceria_lattice(15,15,15)

points = bob[['x','y','z']].values
particle_hull  = ob_gen.hull_from_points(points)
particle_hull = ob_gen.expand_hull(particle_hull,0.000000001)

support_hull = ob_gen.support_hull(8,50,50,particle_interface_points=np.array([[5.69638606, 2.88445904, 0],[0.30238439, 6.99237546, 0],[-4.64750397, 4.79482633, 0],[-5.62285321, -1.09368618, 0],[-1.00215511, -4.9157151, 0],[5.76586847, -0.65974722, 0]]))
print(particle_hull[0].points,support_hull)
particle_new, support_new = ob_gen.add_step(particle_hull.copy(),support_hull.copy())
ob_gen.visualize_hull_points(particle_new)
ob_gen.visualize_hull_points(particle_hull)
filtered_Pt=ob_gen.filter_atoms_by_hull(bob,particle_new)
#relaxed_struct = ob_gen.relax_structure(filtered_Pt)
#ob_gen.mayavi_atomic_structure(relaxed_struct)
print("Filtered Pt",filtered_Pt)   
filtered_ceo2 = ob_gen.filter_atoms_by_hull(CeO2_bulk_test,support_new)
filtered_atoms = pd.concat([filtered_Pt,filtered_ceo2])
relaxed_struct = ob_gen.set_interface_spacing(filtered_atoms, 2.1)
relaxed_struct = ob_gen.relax_structure(relaxed_struct)
#ob_gen.mayavi_atomic_structure(relaxed_struct)
filtered_atoms.label = filtered_atoms.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
ob_gen.mayavi_atomic_structure(filtered_atoms)
# %%
relaxed_struct = ob_gen.remove_overlapping_atoms(relaxed_struct)
relaxed_struct.label = relaxed_struct.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
ob_gen.mayavi_atomic_structure(relaxed_struct)
# %%
def save_as_xyz(df, file_name):
    """
    Save a DataFrame as an XYZ file.

    Parameters:
    df (DataFrame): DataFrame containing columns 'x', 'y', 'z', and 'label'.
    file_name (str): Name of the XYZ file to be saved.
    """
    with open(file_name, 'w') as file:
        # Write the number of atoms as the first line
        file.write(f'{len(df)}\n')
        # Write a comment line - can be empty or descriptive
        file.write('XYZ file generated from DataFrame\n')
        # Write atom positions
        for index, row in df.iterrows():
            file.write(f'{row["label"]} {row["x"]} {row["y"]} {row["z"]}\n')

# Example usage with the provided DataFrame

relaxed_struct.label = relaxed_struct.label.replace({0: 'Ce', 1: 'O', 2: 'Pt'})
save_as_xyz(relaxed_struct, 'atom_data.xyz')
# %%
