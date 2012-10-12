import numpy as np
from IPython.parallel import Client
from IPython.parallel.util import interactive
from dot_layer_solver.layer_tree import Layer, LayerTree

rc = Client()
dview = rc[:]

alpha=2.7439
    
dview.execute('alpha=2.7439',block=True)
dview.execute('tree=None',block=True)
dview.execute('e=None',block=True)
dview.execute('layers=[]',block=True)
dview.execute('context=None',block=True)


dview.execute('import sys',block=True)
dview.execute("sys.path.append('/Users/betcke/development/dot_layered_media_solver')",block=True)
dview.execute("sys.path.append('/Users/betcke/local/bempp/python')",block=True)
dview.execute("from dot_layer_solver.dot_backend import *",block=True)
dview.execute("from dot_layer_solver.layer_tree import *",block=True)

@dview.remote(block=True)
@interactive
def initializeContext():
    from dot_layer_solver.dot_backend import initializeContext
    global context
    context = initializeContext()

@dview.remote(block=True)
@interactive
def addLayer(mesh,mua,mus,omega,c,layer_id):
    global layers
    layers.append(Layer(mesh,mua,mus,omega,c,layer_id))
    
@dview.remote(block=True)
@interactive
def initializeTree():
    global tree
    tree = LayerTree(layers[0])

@dview.remote(block=True)
@interactive
def addTreeLayer(inner,outer):
    global tree
    tree.add_layer(layers[inner],layers[outer])
    
@dview.parallel(block=True)
@interactive
def initEvaluators(layer_id):
    from dot_layer_solver.dot_backend import Evaluator
    global e
    e = Evaluator(tree,layers[layer_id],context,alpha)

@dview.remote(block=True)
@interactive
def getDofs():
    return e.dofs
            
                
class dot_operator(object):
    
    def __init__(self):
        
        self.alpha = alpha
        self.rc=rc
         
        self.initializeEvaluators()
        self.initializeDimensions()
        
    def initializeEvaluators(self):
        pass
        
    def initializeDimensions(self):
                   
        dims = getDofs()
        self.dimensions=[dims[i][i]['dirichlet']+dims[i][i]['neumann'] for i in self.rc.ids]
        self.dimensions = np.cumsum(self.dimensions)        
        self.dimensions = np.insert(self.dimensions,0,0)
        self.shape = (2*self.dimensions[-1],2*self.dimensions[-1])
        
    def matvec(self,x):
        
        @dview.remote(block=True)
        def remoteMatVec(data):
            return e.apply(data)
        
        dim = self.dimensions[-1]
        if len(x.shape) ==1:
            shape = (dim,)
        else:
            shape = (dim,1)
        xc = np.zeros(shape,dtype='complex128')
        xc.flat=x[:dim]+1j*x[dim:]
        data = {}
        result = np.zeros(xc.shape,dtype='complex128')
        for i in range(len(self.dimensions)-1):
            data[i]=xc.flat[self.dimensions[i]:self.dimensions[i+1]]
            data[i].shape=(len(data[i]),1)
        result_data = remoteMatVec(data)
        for i in self.rc.ids:
            result.flat[self.dimensions[i]:self.dimensions[i+1]]+=result_data[i].flat
        return np.vstack([np.real(result),np.imag(result)])
    
#    def initializeAcaPreconditioner(self,accuracy=1E-3):
#        
#        for e in self.evaluators: e.initializeAcaPreconditioner(accuracy)
#    
#    def applyAcaPreconditioner(self,x):
#
#        dim = self.dimensions[-1]
#        if len(x.shape) ==1:
#            shape = (dim,)
#        else:
#            shape = (dim,1)
#        xc = np.zeros(shape,dtype='complex128')
#        xc.flat=x[:dim]+1j*x[dim:]
#        data = {}
#        result = np.zeros(xc.shape,dtype='complex128')
#        for i in range(len(self.dimensions)-1):
#            data[i]=xc.flat[self.dimensions[i]:self.dimensions[i+1]]
#            data[i].shape=(len(data[i]),1)
#        for i in range(len(self.evaluators)):
#            result.flat[self.dimensions[i]:self.dimensions[i+1]]+=self.evaluators[i].applyAcaPreconditioner(data).flat
#        return np.vstack([np.real(result),np.imag(result)])
        

    def getRhs(self,evalFun):
        
        zero_view = self.rc[0]
        @zero_view.remote(block=True)
        @interactive
        def reallyGetRhs(evalFun):
            return e.getRhs(evalFun)
        
        data = reallyGetRhs(evalFun)
        rhs = np.zeros((self.dimensions[-1],1),dtype='complex128')
        rhs[0:len(data),0]=data
        rhs_real = np.real(rhs)
        rhs_imag = np.imag(rhs)
        return np.vstack([rhs_real,rhs_imag])
        
        
        #    def getRhs(self,evalFun):
#        
#        # Create the mesh on the outer domain
#        
#        root = self.layer_tree.get_root()
#        grid_factory = createGridFactory()
#        root_grid = grid_factory.importGmshGrid("triangular",root.mesh)
#        root_space_neumann = createPiecewiseConstantScalarSpace(self.context,root_grid)
#        root_space_dirichlet = createPiecewiseLinearContinuousScalarSpace(self.context,root_grid)
#        fun = createGridFunction(self.context,root_space_neumann,root_space_dirichlet,evalFun)
#        
#        data = fun.projections()
#        rhs = np.zeros((self.dimensions[-1],1),dtype='complex128')
#        rhs[0:len(data),0]=data
#        rhs_real = np.real(rhs)
#        rhs_imag = np.imag(rhs)
#        return np.vstack([rhs_real,rhs_imag])

        
        
        
        
class AcaPreconditioner(object):
    
    def __init__(self,dotOperator,accuracy=1E-3):
        self.dotOperator = dotOperator
        self.shape = dotOperator.shape
        dotOperator.initializeAcaPreconditioner(accuracy)
        
    def matvec(self,x):
        
        return self.dotOperator.applyAcaPreconditioner(x)
    
    
        
            
            
            
       
        
            
            
        
        
        
    
                
        
        
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
            
            
        
            
    
        
            
            
            
            
        
        
        
        
        