from PyTrilinos import Epetra
from bempp import lib,shapes

import numpy as np

def compute_wavenumbers(mu_absorption,mu_scattering,c,omega):
    """Compute a vector of complex wavenumbers from the vectors
       mu_absoprtion and mu_scattering.
    """

    n = len(mu_absorption)
    kappa = [1./(3.*(mu_absorption[i]+mu_scattering[i])) for i in range(n)]
    wavenumbers = [np.sqrt(mua[i]/kappa[i] + 1j*omega/(c*kappa[i])) for i in range(n)]

    return wavenumbers


def initialize_context():
    
    accuracyOptions = lib.createAccuracyOptions()
    accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
    quadStrategy = lib.createNumericalQuadratureStrategy("float64","complex128",accuracyOptions)
    options = lib.createAssemblyOptions()
    options.switchToAcaMode(lib.createAcaOptions())
    context = lib.createContext(quadStrategy,options)
    return context

def initialize_spaces(context,grid):
    """Initialize the spaces of piecewise linear and piecewise constant functions
       for a given grid

       Return a dictionary
        {'l': Piecewise linear continuous space
         'c': Piecewise constant space
         'ndofl': Number of dofs for lin space
         'ndofc': Number of dofs for const space
         'ndof': Number of total dofs (ndofc+ndofl)
        }  
    """

    res = {}
    res['l'] = lib.createPiecewiseLinearContinuousScalarSpace(context,grid)
    res['c'] = lib.createPiecewiseConstantScalarSpace(context,grid)
    res['ndofl'] = res['l'].globalDofCount()
    res['ndofc'] = res['c'].globalDofCount()
    res['ndof'] = res['ndofl']+res['ndofc']

    return res

def initialize_layers(context, graph, layer_grids, wavenumbers):
    """Returns an array of layers. Each layer is represented by
       a dictionary of the folling form
       
       {'k': wavenumber
        'spaces': dictionary containing the space information for the layer
        'sons': array of son ids
        'father': father id
       }
    """


    def find_father(elem,layers):
        for j in graph[elem]:
            layers[j]['father'] = elem
            if graph.has_key(j):
                find_father(j,layers)
        

    nlayers = len(layer_grids)
    layers = nlayers*[None]
    
    for i in range(nlayers):
        layers[i] = {}
        layers[i]['k'] = wavenumbers[i]
        layers[i]['spaces'] = initialize_spaces(context,layer_grids[i])
        if graph.has_key(i):
            layers[i]['sons'] = graph[i]
        else:
            layers[i]['sons'] = None
    find_father(0,layers)
    layers[0]['father'] = None
    return layers

def diagonal_block(context,layers,layer_id,alpha):
    """Create a diagonal block associated with a given layer_id
    
    """

    plc = layers[layer_id]['spaces']['l']
    pwc = layers[layer_id]['spaces']['c']
    k = layers[layer_id]['k']

    if layer_id == 0:
        I_00  = lib.createIdentityOperator(context,plc,pwc,plc)
        I_01 = lib.createIdentityOperator(context,pwc,pwc,plc)
        I_10 = lib.createIdentityOperator(context,plc,plc,pwc)
        K = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,k)
        S = 1./k*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,k)

        return lib.createBlockedBoundaryOperator(context,[[I_00,2*alpha*I_01],[-.5*I_10-K,S]])
    else:
        kf = layers[layers[layer_id]['father']]['k']
        DD = (-kf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc,pwc,plc,kf)
              -k*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc,pwc,plc,k))
        TT = (-lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc,pwc,plc,kf)
              -lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc,pwc,plc,k))
        KK = (lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,kf)
              +lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,k))
        SS = (-1./kf*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,kf)
              -1./k*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,k))
        
        return lib.createBlockedBoundaryOperator(context,[[DD,TT],[KK,SS]])

def off_diagonal_block_father_son(context,layers,father_id,son_id):
    """Off-diagonal interaction between two different layers. Returns a tuple of two operators.
    """

    plc_father = layers[father_id]['spaces']['l']
    pwc_father = layers[father_id]['spaces']['c']
    plc_son = layers[son_id]['spaces']['l']
    pwc_son = layers[son_id]['spaces']['c']

    kf = layers[father_id]['k']
    
    if father_id==0:
        null_00 = lib.createNullOperator(context,plc_son,pwc_father,plc_father)
        null_01 = lib.createNullOperator(context,pwc_son,pwc_father,plc_father)
        Kf = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_son,plc_father,pwc_father,kf)
        Sf = 1./kf*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc_son,plc_father,pwc_father,kf)

        op_father_son = lib.createBlockedBoundaryOperator(context,[[null_00,null_01],[Kf,-Sf]])

        Ds = kf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_father,pwc_son,plc_son,kf)
        Ts = lib.adjoint(Kf,pwc_son)
        Ks = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_father,plc_son,pwc_son,kf)
        Ss = lib.adjoint(Sf,plc_son)

        op_son_father = lib.createBlockedBoundaryOperator(context,[[Ds,Ts],[-Ks,Ss]])

    else:
        Df = kf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_son,pwc_father,plc_father,kf)
        Tf = lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc_son,pwc_father,plc_father,kf)
        Kf = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_son,plc_father,pwc_father,kf)
        Sf = 1./kf*lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,pwc_son,plc_father,pwc_father,kf)

        op_father_son = lib.createBlockedBoundaryOperator(context,[[Df,Tf],[-Kf,Sf]])

        Ds = lib.adjoint(Df,pwc_son)
        Ts = lib.adjoint(Kf,pwc_son)
        Ks = lib.adjoint(Tf,plc_son)
        Ss = lib.adjoint(Sf,plc_son)

        op_son_father = lib.createBlockedBoundaryOperator(context,[[Ds,Ts],[-Ks,Ss]])

    return (op_father_son,op_son_father)
        
def off_diagonal_block_same_layer(context,layers,id1,id2):
    """Off-diagonal interaction between two different layers. Returns a tuple of two operators.
    """

    pass
        

    

    
    
    

if __name__=="__main__":
    
    mua = [.01, .02, .03]
    mus = [1., .5, .25]
    c = .3
    freq = 100e6
    omega = 2*np.pi*freq*1E-12
    wavenumbers = compute_wavenumbers(mua,mus,c,omega)


    graph = {0:[1],1:[2]}
    radii = [1.5, 2.5, 3.0]
    layer_grids = [shapes.sphere(radii[i],h=.3) for i in range(len(radii))]

    context = initialize_context()
    layers = initialize_layers(context,graph,layer_grids,wavenumbers)
    


    

    
