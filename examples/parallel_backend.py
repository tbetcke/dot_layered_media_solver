import sys
sys.path.append('/Users/betcke/local/bempp/python')
sys.path.append('/Users/betcke/development/dot_layered_media_solver')

from dot_layer_solver.layer_tree import Layer, LayerTree
from dot_layer_solver.dot_evaluation import Evaluator, initializeContext

alpha = 2.7439

e = None

# Read in the grid files

layers = []
tree = None

context = initializeContext()

def initEvaluatorsEngine(layer_id):
    global e
    e = Evaluator(tree,layers[layer_id],context,alpha)
    


