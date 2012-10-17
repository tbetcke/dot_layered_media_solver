import numpy as np
from bempp.lib import *

def initializeContext():
    
    accuracyOptions = createAccuracyOptions()
    accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
    quadStrategy = createNumericalQuadratureStrategy("float64","complex128",accuracyOptions)
    options = createAssemblyOptions()
    options.switchToAcaMode(createAcaOptions())
    context = createContext(quadStrategy,options)
    return context

class Evaluator(object):
    
    def __init__(self,layer_tree,layer,context,alpha):
        self.layer_tree = layer_tree
        self.layer = layer
        self.spaces = {}
        self.dofs = {}
        self.context = context
        self.alpha = alpha
        self.origin = layer_tree.get_origin(layer)
        self.children = layer_tree.get_children(layer)
        self.operators = {}
        self.init_spaces()
        self.initializeOperators()
        
    
    def init_spaces(self):
        
        grid_factory = createGridFactory()    
                
        if self.origin is not None:            
            grid_origin = grid_factory.importGmshGrid("triangular",self.origin.mesh)
            self.spaces[self.origin.id] = {}
            self.spaces[self.origin.id]['dirichlet'] = createPiecewiseLinearContinuousScalarSpace(self.context,grid_origin)
            self.spaces[self.origin.id]['neumann'] = createPiecewiseConstantScalarSpace(self.context,grid_origin)

            for child in self.layer_tree.get_children(self.origin):
                grid_child = grid_factory.importGmshGrid("triangular",child.mesh)
                self.spaces[child.id]={}
                self.spaces[child.id]['dirichlet'] = createPiecewiseLinearContinuousScalarSpace(self.context,grid_child)
                self.spaces[child.id]['neumann'] = createPiecewiseConstantScalarSpace(self.context,grid_child)
        
        # The child layers
        
        else:
            grid_layer = grid_factory.importGmshGrid("triangular",self.layer.mesh)
            self.spaces[self.layer.id] = {}
            self.spaces[self.layer.id]['dirichlet'] = createPiecewiseLinearContinuousScalarSpace(self.context,grid_layer)
            self.spaces[self.layer.id]['neumann'] = createPiecewiseConstantScalarSpace(self.context,grid_layer)
                
        for child in self.children:
            grid_child = grid_factory.importGmshGrid("triangular",child.mesh)
            self.spaces[child.id]={}
            self.spaces[child.id]['dirichlet'] = createPiecewiseLinearContinuousScalarSpace(self.context,grid_child)
            self.spaces[child.id]['neumann'] = createPiecewiseConstantScalarSpace(self.context,grid_child)

        for layer_id in self.spaces:
            self.dofs[layer_id] = {}
            self.dofs[layer_id]['dirichlet'] = self.spaces[layer_id]['dirichlet'].globalDofCount()
            self.dofs[layer_id]['neumann'] = self.spaces[layer_id]['neumann'].globalDofCount()
            
            
    def initializeOperators(self):
        
        layer_id = self.layer.id
        layer_space = self.spaces[layer_id]
        
        
        # The diagonal block operator
        
        
        if self.origin is not None:
            
            origin_id = self.origin.id        
            origin_space = self.spaces[origin_id]


            ko = self.origin.kappa
            kl = self.layer.kappa 

        
            D1 = ko*createModifiedHelmholtz3dHypersingularBoundaryOperator(self.context,layer_space['dirichlet'],
                                                                        layer_space['neumann'],layer_space['dirichlet'],self.origin.k)
            D2 = kl*createModifiedHelmholtz3dHypersingularBoundaryOperator(self.context,layer_space['dirichlet'],
                                                                        layer_space['neumann'],layer_space['dirichlet'],self.layer.k)
            
            K1 = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,layer_space['dirichlet'],
                                                                        layer_space['dirichlet'],layer_space['neumann'],self.origin.k)
            K2 = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,layer_space['dirichlet'],
                                                                        layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
            
            T1 = createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(self.context,layer_space['neumann'],
                                                                        layer_space['neumann'],layer_space['dirichlet'],self.origin.k)
            T2 = createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(self.context,layer_space['neumann'],
                                                                        layer_space['neumann'],layer_space['dirichlet'],self.layer.k)
            
            S1 = 1./ko*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,layer_space['neumann'],
                                                                        layer_space['dirichlet'],layer_space['neumann'],self.origin.k)
            S2 = 1./kl*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,layer_space['neumann'],
                                                                        layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
            
            structure = createBlockedOperatorStructure(self.context)
            structure.setBlock(0,0,-D1-D2)
            structure.setBlock(0,1,-T1-T2)
            structure.setBlock(1,0,K1+K2)
            structure.setBlock(1,1,-S1-S2)
            
            self.operators[self.layer.id]= createBlockedBoundaryOperator(self.context,structure)
            
            
            D = ko*createModifiedHelmholtz3dHypersingularBoundaryOperator(self.context,origin_space['dirichlet'],layer_space['neumann'],
                                                                           layer_space['dirichlet'],self.origin.k)
            
            T = createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(self.context,origin_space['neumann'],layer_space['neumann'],
                                                                            layer_space['dirichlet'],self.origin.k)
            
            K = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,origin_space['dirichlet'],layer_space['dirichlet'],
                                                                     layer_space['neumann'],self.origin.k)
            
            S = 1./ko*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,origin_space['neumann'],layer_space['dirichlet'],
                                                                           layer_space['neumann'],self.origin.k)
            
            structure = createBlockedOperatorStructure(self.context)
            structure.setBlock(0,0,D)
            structure.setBlock(0,1,T)
            structure.setBlock(1,0,-1.*K)
            structure.setBlock(1,1,S)
            self.operators[self.origin.id] = createBlockedBoundaryOperator(self.context,structure)
            
            for child in self.children:
                
                child_space = self.spaces[child.id]
                kc = child.kappa
                
                D = kc*createModifiedHelmholtz3dHypersingularBoundaryOperator(self.context,child_space['dirichlet'],layer_space['neumann'],layer_space['dirichlet'],self.layer.k)
                T = createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(self.context,child_space['neumann'],layer_space['neumann'],layer_space['dirichlet'],self.layer.k)
                K = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,child_space['dirichlet'],layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
                S = 1./kc*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,child_space['neumann'],layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
                
                structure = createBlockedOperatorStructure(self.context)
                structure.setBlock(0,0,D)
                structure.setBlock(0,1,T)
                structure.setBlock(1,0,-K)
                structure.setBlock(1,1,S)
                
                self.operators[child.id] = createBlockedBoundaryOperator(self.context,structure)
        else:
            
            kl = self.layer.kappa
            
            I00 = createIdentityOperator(self.context,layer_space['dirichlet'],layer_space['neumann'],layer_space['dirichlet'])
            I01 = createIdentityOperator(self.context,layer_space['neumann'],layer_space['neumann'],layer_space['dirichlet'])
            I10 = createIdentityOperator(self.context,layer_space['dirichlet'],layer_space['dirichlet'],layer_space['neumann'])
            K   = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,layer_space['dirichlet'],layer_space['dirichlet'],
                                                                       layer_space['neumann'],self.layer.k)
            S   = 1./kl*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,layer_space['neumann'],
                                                                             layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
            
            structure = createBlockedOperatorStructure(self.context)
            structure.setBlock(0,0,I00)
            structure.setBlock(0,1,2*self.alpha*I01)
            structure.setBlock(1,0,-.5*I10-K)
            structure.setBlock(1,1,S)
            
            
            self.operators[self.layer.id]= createBlockedBoundaryOperator(self.context,structure)
            
            for child in self.children:
                
                child_space = self.spaces[child.id]
                kc = child.kappa
                
                N1 = createNullOperator(self.context,child_space['dirichlet'],layer_space['neumann'],layer_space['dirichlet'])
                N2 = createNullOperator(self.context,child_space['neumann'],layer_space['neumann'],layer_space['dirichlet'])
                K = createModifiedHelmholtz3dDoubleLayerBoundaryOperator(self.context,child_space['dirichlet'],layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
                S = 1./kc*createModifiedHelmholtz3dSingleLayerBoundaryOperator(self.context,child_space['neumann'],layer_space['dirichlet'],layer_space['neumann'],self.layer.k)
                
                structure = createBlockedOperatorStructure(self.context)
                structure.setBlock(0,0,N1)
                structure.setBlock(0,1,N2)
                structure.setBlock(1,0,K)
                structure.setBlock(1,1,-S)
                
                self.operators[child.id] = createBlockedBoundaryOperator(self.context,structure)
                

    def getRhs(self,evalFun):
                
        layer_id = self.layer.id
        fun = createGridFunction(self.context,self.spaces[layer_id]['neumann'],self.spaces[layer_id]['dirichlet'],evalFun)        
        data = fun.projections()
        return data
  
    def saveResult(self,coefficient_data):
        
        layer_id = self.layer.id
        
        coeffs_dirichlet = coefficient_data[layer_id][:self.dofs[layer_id]['dirichlet']].flat
        coeffs_neumann = coefficient_data[layer_id][self.dofs[layer_id]['dirichlet']:].flat
        gridFun_dirichlet = createGridFunction(self.context,self.spaces[layer_id]['dirichlet'],self.spaces[layer_id]['neumann'],coefficients=coeffs_dirichlet)
        gridFun_neumann = createGridFunction(self.context,self.spaces[layer_id]['neumann'],self.spaces[layer_id]['dirichlet'],coefficients=coeffs_neumann)        
        gridFun_neumann.exportToVtk("cell_data","neumann_data","v"+str(layer_id))
        gridFun_dirichlet.exportToVtk("cell_data","dirichlet_data","u"+str(layer_id))
        
                
    def apply(self,data):

        
        result = np.zeros((self.dofs[self.layer.id]['dirichlet']+self.dofs[self.layer.id]['neumann'],1),dtype='complex128')
        
        for block_id in self.operators:
            op = self.operators[block_id].weakForm()
            result = result + op*data[block_id]
        
        return result
    
    def initializeAcaPreconditioner(self,accuracy=1E-3):
        print "Initialize Preconditioner for Layer %i" % self.layer.id
        if self.layer.id == 0:
            m00 = discreteSparseInverse(self.operators[0].block(0,0).weakForm())
            m11 = acaOperatorApproximateLuInverse(self.operators[0].block(1,1).weakForm().asDiscreteAcaBoundaryOperator(),accuracy)
            print "After the ACA Operation on layer 0"
            self.precond = [m00,m11]
            print "Preconditioner created on layer 0"
        else:
            m00 = acaOperatorApproximateLuInverse(self.operators[self.layer.id].block(0,0).weakForm().asDiscreteAcaBoundaryOperator(),accuracy)
            m11 = acaOperatorApproximateLuInverse(self.operators[self.layer.id].block(1,1).weakForm().asDiscreteAcaBoundaryOperator(),accuracy)
            self.precond = [m00,m11]
                        
    def applyAcaPreconditioner(self,data):
        
        localData = data[self.layer.id]
        offsets = np.insert(np.cumsum([self.precond[0].rowCount(),self.precond[1].rowCount()]),0,0)
        result = np.zeros(localData.shape,dtype='complex128')
        result[offsets[0]:offsets[1],:] = self.precond[0]*localData[offsets[0]:offsets[1],:]
        result[offsets[1]:offsets[2],:] = self.precond[1]*localData[offsets[1]:offsets[2],:]
        
        return result
