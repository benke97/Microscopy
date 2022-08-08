import numpy as np


def write_PSLG(name,phase,edge_loop_idx):
    
    #----------------------------Generate PSLG-----------------------------------
    f = open(str(name) + ".poly", "w")
    #-----
    f.write("# " + str(name) + ".poly" + '\n')
    f.write("#" + '\n')
    f.write("# A" + str(name) + "particle with X points in 2D, no attributes, one boundary marker." + '\n')
    f.write(str(len(phase)) + " 2" + " 0" + " 1" + '\n')
    f.write("# Perimeter" + '\n')

    for j in phase:
        atom_idx = np.where(np.all(phase==j,axis=1))
        print(atom_idx[0])
        atom_idx = atom_idx[0]
        if atom_idx in edge_loop_idx:
            f.write(str(int(atom_idx+1)) + " " + str(phase[atom_idx][0,0]) + " " + str(phase[atom_idx][0,1]) + " 2" + '\n')
        else:
            f.write(str(int(atom_idx+1)) + " " + str(phase[atom_idx][0,0]) + " " + str(phase[atom_idx][0,1]) + " 0" + '\n')

    #for i in edge_loop_idx:
        #print(platinum[i])
        #f.write(str(edge_loop_idx.index(i)+1) + " " + str(platinum[i][0,0]) + " " + str(platinum[i][0,1]) + " 2" + '\n')
    f.write('\n')
    #-----
    f.write("# X segments, each with boundary marker." + '\n')
    f.write(str(len(edge_loop_idx)) + " 1" + '\n')
    f.write("# Perimeter" + '\n')

    for j in edge_loop_idx:
        vertex_idx = edge_loop_idx.index(j)
        if vertex_idx+1 == len(edge_loop_idx):
            next_value = edge_loop_idx[0]
        else:
            print(vertex_idx,len(edge_loop_idx))
            print(int(vertex_idx+1))
            next_value = edge_loop_idx[int(vertex_idx+1)] 
        f.write(str(int(vertex_idx+1)) + " " + str(int(j+1)) + " " + str(int(next_value+1)) + " 2" + '\n')





    #for i in edge_loop_idx:
    #    if edge_loop_idx.index(i) + 1 < len(edge_loop_idx):
    #        f.write(str(edge_loop_idx.index(i)+1) + " " + str(edge_loop_idx.index(i)+1) + " " + str((edge_loop_idx.index(i)+2)%(len(edge_loop_idx)+1)) + " 2" + '\n')
    #    else:
    #        f.write(str(edge_loop_idx.index(i)+1) + " " + str(edge_loop_idx.index(i)+1) + " " + " 1" + " 2" + '\n')
    f.write('\n')
    #-----
    f.write("# No holes" + '\n')
    f.write("0")

    f.close()