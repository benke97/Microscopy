#%%
import pandas as pd
import numpy as np
import os
import pickle as pkl
from ObjectGenerator import ObjectGenerator
from ParameterRandomizer import ParameterRandomizer
#%%
class DatasetGenerator():
    def __init__(self,save_path):
        self.save_path = save_path
        self.obj_gen = ObjectGenerator()
        self.ParameterRandomizer = ParameterRandomizer()

    def add_to_dict(self,dict_of_parameters,dict_to_add):
        for key in dict_to_add:
            dict_of_parameters[key] = dict_to_add[key]
        return dict_of_parameters

    def save_dataset(self,dataset):
        pkldir = os.path.join(self.save_path, "pkl")
        os.makedirs(pkldir, exist_ok=True)
        pklpath = os.path.join(pkldir, "dataset.pkl")

        with open(pklpath, "wb") as file:
            pkl.dump(dataset, file)
        print("Dataset saved to", pklpath)

    def generate_dataset(self,num_structures, composition, support_layers=8, support_depth=50, support_width=50, CeO2_bulk_depth=20, CeO2_bulk_width=20, CeO2_bulk_height=20, Pt_bulk_depth=10, Pt_bulk_width=10, Pt_bulk_height=10, wulff_element="Pt", wulff_rounding="above", cluster_element="Pt"):
        """
        num_structures: The number of structures to generate
        Composition = [num_random,num_wulff,num_cluster] """
        assert sum(composition) == num_structures, "The sum of the composition must equal the number of structures"
        dfs = []
        for i in range(num_structures):
            print((i))
            if composition[0] > 0: # Random structure
                dict_of_parameters = self.ParameterRandomizer.randomize_parameters("random")
                dict_of_parameters = self.add_to_dict(dict_of_parameters,{"support_layers":support_layers,
                                                                            "support_depth":support_depth,
                                                                            "support_width":support_width,
                                                                            "CeO2_bulk_depth":CeO2_bulk_depth,
                                                                            "CeO2_bulk_width":CeO2_bulk_width,
                                                                            "CeO2_bulk_height":CeO2_bulk_height,
                                                                            "Pt_bulk_depth":Pt_bulk_depth,
                                                                            "Pt_bulk_width":Pt_bulk_width,
                                                                            "Pt_bulk_height":Pt_bulk_height})
                structure_df = self.obj_gen.generate_atomic_structure("random",dict_of_parameters)
                structure_df["structure_type"] = "random"
                dfs.append(structure_df)
            if composition[1] > 0: # Wulff structure
                dict_of_parameters = self.ParameterRandomizer.randomize_parameters("wulff")
                print("wulff size",dict_of_parameters["wulff_size"])
                dict_of_parameters = self.add_to_dict(dict_of_parameters,{"support_layers":support_layers,
                                                                            "support_depth":support_depth,
                                                                            "support_width":support_width,
                                                                            "CeO2_bulk_depth":CeO2_bulk_depth,
                                                                            "CeO2_bulk_width":CeO2_bulk_width,
                                                                            "CeO2_bulk_height":CeO2_bulk_height,
                                                                            "wulff_element":wulff_element,
                                                                            "wulff_rounding":wulff_rounding})
                structure_df = self.obj_gen.generate_atomic_structure("wulff",dict_of_parameters)
                structure_df["structure_type"] = "wulff"
                dfs.append(structure_df)
                visu = structure_df.copy()
                visu.label = visu.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
                #self.obj_gen.mayavi_atomic_structure(visu)
            
            if composition[2] > 0: # Cluster structure
                dict_of_parameters = self.ParameterRandomizer.randomize_parameters("cluster")
                dict_of_parameters = self.add_to_dict(dict_of_parameters,{"support_layers":support_layers,
                                                                            "support_depth":support_depth,
                                                                            "support_width":support_width,
                                                                            "CeO2_bulk_depth":CeO2_bulk_depth,
                                                                            "particle_support_facet":"111",
                                                                            "CeO2_bulk_width":CeO2_bulk_width,
                                                                            "CeO2_bulk_height":CeO2_bulk_height,
                                                                            "cluster_element":cluster_element})
                structure_df = self.obj_gen.generate_atomic_structure("cluster",dict_of_parameters)
                structure_df["structure_type"] = "cluster"
                dfs.append(structure_df)

        dataset = pd.concat(dfs,ignore_index=True)
        self.save_dataset(dataset)
#%%
data_gen = DatasetGenerator(os.getcwd())
data_gen.generate_dataset(100,[0,100,0])

#%%
