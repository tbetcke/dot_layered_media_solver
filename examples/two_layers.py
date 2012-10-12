import sys
sys.path.append('/Users/betcke/local/bempp/python')
sys.path.append('/Users/betcke/development/dot_layered_media_solver')

import numpy as np
from bempp import lib as blib
import tempfile
import os
import subprocess
from IPython.parallel import Client
import dot_layer_solver.dot_evaluation as dot

def evalBoundaryData(point):
    return 1

def evalNullData(point):
    return 0


# Define physical parameters

c = .3
freq = 100e6
omega = 2*np.pi*freq*1E-12
alpha = 2.7439

# Outer region

mua1 = .01
mus1 = 1.
kappa1 = 1./(3.*(mua1+mus1))
w1 = np.sqrt(mua1/kappa1+1j*omega/(c*kappa1))

# Inner region
mua2 = .02
mus2 = .5
kappa2 = 1./(3.*(mua2+mus2))
w2 = np.sqrt(mua2/kappa2+1j*omega/(c*kappa2))


# We consider two spheres. One has radius r1, the other radius r2<r1. We want to look at
# the low-rank interaction between the spheres. Gmsh is used to create the meshes

r1 = 2.5
r2 = 1.5
element_size = .5
gmsh_command = "/Applications/Gmsh.app/Contents/MacOS/gmsh"
sphere_definition = "sphere.txt"

sphere_def = open(sphere_definition,'r').read()

# Construct two Gmsh meshes with the required parameters

s1_geo, s1_geo_name = tempfile.mkstemp(suffix='.geo',dir=os.getcwd(),text=True)
s2_geo, s2_geo_name = tempfile.mkstemp(suffix='.geo',dir=os.getcwd(),text=True)
s1_msh_name = os.path.splitext(s1_geo_name)[0]+".msh"
s2_msh_name = os.path.splitext(s2_geo_name)[0]+".msh"

s1_geo_f = os.fdopen(s1_geo,"w")
s2_geo_f = os.fdopen(s2_geo,"w")

s1_geo_f.write("rad = "+str(r1)+";\nlc = "+str(element_size)+";\n"+sphere_def)
s2_geo_f.write("rad = "+str(r2)+";\nlc = "+str(element_size)+";\n"+sphere_def)
s1_geo_f.close()
s2_geo_f.close()

# Use Gmsh to create meshes

subprocess.check_call(gmsh_command+" -2 "+s1_geo_name,shell=True)
subprocess.check_call(gmsh_command+" -2 "+s2_geo_name,shell=True)

# Read the meshes into BEM++ Objects

#sphere1 = blib.createGridFactory().importGmshGrid("triangular",s1_msh_name)
#sphere2 = blib.createGridFactory().importGmshGrid("triangular",s2_msh_name)

# Generate the tree structure

dot.initializeContext()
dot.addLayer(s1_msh_name,0.01,1.,omega,c,0)
dot.addLayer(s2_msh_name,0.02,.5,omega,c,1)
dot.initializeTree()
dot.addTreeLayer(1,0)
dot.initEvaluators.map(dot.rc.ids)



#outer = Layer(s1_msh_name,0.01,1.,omega,c,0)
#inner = Layer(s2_msh_name,0.02,.5,omega,c,1)

#tree = LayerTree(outer)
#tree.add_layer(inner,outer)


operator = dot.dot_operator()
print operator.dimensions
print operator.shape

# Generate a right-hand side


data = operator.getRhs(evalBoundaryData)
print data
#print len(data)
#
#v = np.ones((operator.shape[0],1))
#b = operator.matvec(v)
##print b
#
#prec = AcaPreconditioner(operator)
#b2 = prec.matvec(v)
##print b2
#
## Solve the system
#
#from scipy.sparse.linalg import gmres
#
#x,info = gmres(operator,data,M=prec)
#dim = operator.dimensions[-1]
#result = x[:dim]+1j*x[dim:]
#print result
#print info



# Clean up the temporary files

os.remove(s1_geo_name)
os.remove(s2_geo_name)
os.remove(s1_msh_name)
os.remove(s2_msh_name)
