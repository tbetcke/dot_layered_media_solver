# Data Structures to store trees of grids
import numpy as np

class Layer(object):
    """ Store a layer and the necessary material information """

    def __init__(self,mesh,mua,mus,omega,c,layer_id):
        """Initialize a layer"""

        self.mesh = mesh
        self.kappa = 1./(3*(mua+mus))
        self.mua = mua
        self.mus = mus
        self.omega = omega
        self.c = c
        self.k = np.sqrt(mua/self.kappa+1j*omega/(c*self.kappa))
        self.id = layer_id

class LayerTree(object):
    """ Create a tree structure for layers"""


    def __init__(self,root):
        from collections import namedtuple

        self.TreeElem = namedtuple('TreeElem',['layer','origin','children'])
        self.tree = {root.id:self.TreeElem(root,None,[])}

    def add_layer(self,layer,origin):
        """ Add a layer within an origin layer"""

        if self.tree.has_key(layer.id):
            raise Exception("Layer is already part of the tree")

        self.tree[layer.id] = self.TreeElem(layer,origin,[])
        self.tree[origin.id].children.append(layer)

    def get_origin(self,layer):
        """Return the origin layer"""
        return self.tree[layer.id].origin

    def get_children(self,layer):
        """Return the children sorted by id"""
        children = self.tree[layer.id].children[:]
        children.sort(key=lambda child: child.id)
        return children
    
    def is_root(self,layer):
        return (layer.origin is None)
    
    def get_root(self):
        return self.tree[0].layer
    


