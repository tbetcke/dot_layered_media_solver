from bempp.lib import *
from dot_layer_solver.layer_tree import *
from dot_layer_solver.dot_evaluation import *

# Read in the grid files

grid_factory = createGridFactory()
skin = "../meshes/headsurf1.gmsh"
csf = "../meshes/headsurf2.gmsh"

skin_layer = Layer(skin,0.01,1.,0.2*1E-3,0.3/1.4,0)
csf_layer = Layer(csf,0.005,2.,0.2*1E-3,0.3/1.4,1)

tree = LayerTree(skin_layer)
tree.add_layer(csf_layer,skin_layer)


operator = dot_operator(tree,2.7439)
print operator.dimensions
print operator.shape


#skin_evaluator = Evaluator(tree,skin_layer,context,2.7439)
#skin_evaluator.initializeOperators()
#csf_evaluator = Evaluator(tree,csf_layer,context,2.7439)
#csf_evaluator.initializeOperators()
















