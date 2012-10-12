import sys
sys.path.append('/Users/betcke/local/bempp/python')
sys.path.append('/Users/betcke/development/dot_layered_media_solver')


from dot_layer_solver.layer_tree import *
from dot_layer_solver.dot_evaluation import *

from IPython.parallel import Client
from IPython.parallel.util import interactive

rc = Client()
dview=rc[:]
print rc.ids
res = dview.run('/Users/betcke/development/dot_layered_media_solver/examples/parallel_backend.py',block=True)
print "I am here"

@dview.remote(block=True)
def addLayer(mesh,mua,mus,omega,c,layer_id):
    global layers
    layers.append(Layer(mesh,mua,mus,omega,c,layer_id))
    
@dview.remote(block=True)
def initializeTree():
    global tree
    tree = LayerTree(layers[0])

@dview.remote(block=True)
def addTreeLayer(inner,outer):
    global tree
    tree.add_layer(layers[inner],layers[outer])
    

skin = "/Users/betcke/development/dot_layered_media_solver/meshes/headsurf1.gmsh"
csf = "/Users/betcke/development/dot_layered_media_solver/meshes/headsurf2.gmsh"

addLayer(skin,0.01,1.,0.2*1E-3,0.3/1.4,0)
addLayer(csf,0.005,2.,0.2*1E-3,0.3/1.4,1)
initializeTree()
addTreeLayer(1,0)

@dview.parallel(block=True)
def initEvaluators(layer_id):
    global e
    global tree
    e = Evaluator(tree,layers[layer_id],context,alpha)


res=initEvaluators.map(range(2))
res=dview.execute('globals()',block=True)
print res

