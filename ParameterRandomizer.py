import numpy as np
import random
class ParameterRandomizer():
    def __init__(self):
        pass
    
    def gen_params_random(self):
        # hull layers
        # interface radiis
        # layer sample points
        # centers
        # add_step
        # step_height
        # surface_facet
        # particle_surface_facet
        # particle_rotation
        hull_layers = random.randint(2,7)
        interface_radiis = []
        layer_sample_points = []
        centers = []
        wetting = random.choice(["acute","obtuse","90"])
        if wetting == "90":
            radius = random.randint(4,15)
            for i in range(hull_layers):
                interface_radiis.append(radius)
        
        if wetting == "acute":
            radius = random.randint(hull_layers+2,hull_layers+10)
            for i in range(hull_layers):
                if i == 0:
                    interface_radiis.append(radius)
                    continue
                radius = random.randint(min(radius-1,hull_layers),radius)
                interface_radiis.append(radius)
        
        if wetting == "obtuse":
            thickest_layer = random.randint(0, hull_layers - 1)
            radius_thick = random.randint(hull_layers, hull_layers + 10)

            # Layers below the thickest layer (increasing thickness)
            for i in range(thickest_layer, 0, -1):
                radius = random.randint(hull_layers, radius_thick - 1)
                interface_radiis.insert(0, radius)

            # Thickest layer
            interface_radiis.append(radius_thick)

            # Layers above the thickest layer (decreasing thickness)
            for i in range(thickest_layer + 1, hull_layers):
                radius = random.randint(hull_layers, radius_thick - 1)
                interface_radiis.append(radius)

        
        for i in range(hull_layers):
            layer_sample_points.append(random.randint(4,10))
            centers.append([np.random.normal(0,0.5),np.random.normal(0,0.5)])
        
        add_step = random.choice([True,False])
        step_height = random.randint(1,hull_layers)
        surface_facet = random.choice(["100","100","111","111","random"])
        particle_surface_facet = random.choice(["100","100","111","111","random"])
        rotation = random.choice([True,False])
        if rotation:
            particle_rotation = random.uniform(0,2*np.pi)
        else:
            particle_rotation = 0

        dict_of_parameters = {"hull_layers":hull_layers,
                              "interface_radiis":interface_radiis,
                              "layer_sample_points":layer_sample_points,
                              "centers":centers,
                              "add_step":add_step,
                              "step_height":step_height,
                              "surface_facet":surface_facet,
                              "particle_surface_facet":particle_surface_facet,
                              "particle_rotation":particle_rotation}

        return dict_of_parameters
    
    def gen_params_wulff(self):
        # size
        # add_step
        # step_height
        # surface_facet
        # particle_surface_facet
        # particle_rotation

        size = random.randint(25,325)
        add_step = random.choice([True,False])
        step_height = 2
        surface_facet = random.choice(["100","100","111","111","random"])
        particle_surface_facet = random.choice(["100","111"])
        rotation = random.choice([True,False])
        if rotation:
            particle_rotation = random.uniform(0,2*np.pi)
        else:
            particle_rotation = 0
        dict_of_parameters = {"wulff_size":size,
                              "add_step":add_step,
                              "step_height":step_height,
                              "surface_facet":surface_facet,
                              "particle_surface_facet":particle_surface_facet,
                              "particle_rotation":particle_rotation}
        print(dict_of_parameters)
        return dict_of_parameters
    
    def gen_params_cluster(self):

        cluster_surfaces = [(1, 0, 0), (1, 1, 1), (1, -1, 1)]
        cluster_layers = [random.randint(2,6),random.randint(2,6),random.randint(-1,0)]
        surface_facet = random.choice(["100","100","111","111","random"])
        rotation = random.choice([True,False])
        if rotation:
            particle_rotation = random.uniform(0,2*np.pi)
        else:
            particle_rotation = 0

        add_step = random.choice([True,False])
        step_height = 2

        dict_of_parameters = {"cluster_surfaces":cluster_surfaces,
                                "cluster_layers":cluster_layers,
                                "add_step":add_step,
                                "step_height":step_height,
                                "surface_facet":surface_facet,
                                "particle_rotation":particle_rotation}
        return dict_of_parameters



    def randomize_parameters(self,type):
        if type == "random":
            dict_of_parameters = self.gen_params_random()
        elif type == "wulff":
            dict_of_parameters = self.gen_params_wulff()
        elif type == "cluster":
            dict_of_parameters = self.gen_params_cluster()
        else:
            raise ValueError("Unknown structure type")
        return dict_of_parameters
    