from PyTrilinos import Epetra
from bempp import lib,shapes

import numpy as np

comm = Epetra.PyComm()


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
            layers[i]['sons'] = []
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

    pwc_id1 = layers[id1]['spaces']['c']
    plc_id1 = layers[id1]['spaces']['l']
    pwc_id2 = layers[id2]['spaces']['c']
    plc_id2 = layers[id2]['spaces']['l']

    k = layers[layers[id1]['father']]['k']

    D = k*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_id2,pwc_id1,plc_id1,k)
    T = lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc_id2,pwc_id1,plc_id1,k)
    K = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_id2,plc_id1,pwc_id1,k)
    S = lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc_id2,plc_id1,pwc_id1,k)

    D2 = lib.adjoint(D,pwc_id2)
    T2 = lib.adjoint(K,pwc_id2)
    K2  = lib.adjoint(T,plc_id2)
    S2  = lib.adjoint(S,plc_id2)

    op1 = lib.createBlockedBoundaryOperator(context,[[-D,-T],[K,-S]])
    op2 = lib.createBlockedBoundaryOperator(context,[[-D2,-T2],[K2,-S2]])
    
    return (op1,op2)


def generate_local_operator(context,my_proc,graph,layers,alpha):
    """Generate the local system matrix.
    """

    def copy_to_structure(brow,bcol,block,structure):
        """Small helper routine to copy a 2x2 block at the right position into a structure
        """
        structure.setBlock(2*brow,2*bcol,block.block(0,0))
        structure.setBlock(2*brow,2*bcol+1,block.block(0,1))
        structure.setBlock(2*brow+1,2*bcol,block.block(1,0))
        structure.setBlock(2*brow+1,2*bcol+1,block.block(1,1))

    nproc = comm.NumProc()

    import partitioner as part
    process_map = part.procmap(part.blockmap(graph),nproc)
    scheduler = part.partition(process_map)

    # Initialize structure of local system matrix

    nb = process_map.shape[0] # Block-dimension of global matrix
    structure = lib.createBlockedOperatorStructure(context)
    for i in range(nb):
        plc = layers[i]['spaces']['l']
        pwc = layers[i]['spaces']['c']
        null_op_00 = lib.createNullOperator(context,plc,pwc,plc)
        null_op_11 = lib.createNullOperator(context,pwc,plc,pwc)
        structure.setBlock(2*i,2*i,null_op_00)
        structure.setBlock(2*i+1,2*i+1,null_op_11)

    # Iterate through elements from scheduler and fill up local system matrix
    
    for elem in scheduler[my_proc+1]:
        if elem[0]==elem[1]:
            # Diagonal Block
            block = diagonal_block(context,layers,elem[0],alpha)
            copy_to_structure(elem[0],elem[1],block,structure)
        else:
            if layers[elem[0]]['sons'].count(elem[1]):
                # elem[1] is a son of elem[0]
                block1,block2 = off_diagonal_block_father_son(context,layers,elem[0],elem[1])
            else:
                # elements must be on the same level
                block1,block2 = off_diagonal_block_same_layer(context,layers,elem[0],elem[1])
            copy_to_structure(elem[0],elem[1],block1,structure)
            copy_to_structure(elem[1],elem[0],block2,structure)
    A = lib.createBlockedBoundaryOperator(context,structure)
    return A


    
class LocalOperator(Epetra.Operator):
    """Define a Trilinos operator for the matvec product
    """

    def __init__(self,A,operator_map,distributed_map):
        Epetra.Operator.__init__(self)
        self.__label = "LocalOperator"
        self.A = A

        from bempp import tools
        self.operator = tools.RealOperator(A.weakForm()) # Here the weak form is assembled
        self.__operator_map = operator_map
        self.__distributed_map = distributed_map
        self.__comm = comm
        self.__useTranspose = False

    def Label(self):
        return self.__label

    def OperatorDomainMap(self):
        return self.__operator_map

    def OperatorRangeMap(self):
        return self.__operator_map

    def Comm(self):
        return self.__comm

    def ApplyInverse(self):
        return -1

    def HasNormInf(self):
        return False

    def NormInf(self):
        return -1

    def SetUseTranspose(self, useTranspose):
        return -1

    def UseTranspose(self):
        return self__useTranspose

    def Apply(self,x,y):
        try:
            # First import the vector into the distributed_map

            importer = Epetra.Import(self.__distributed_map,self.__operator_map)
            my_xvec = Epetra.MultiVector(self.__distributed_map,x.NumVectors())
            my_xvec.Import(x,importer,Epetra.Insert)

            # Now apply the local operator
            y_data = self.operator.matmat(my_xvec.T)
 
            y_vec = Epetra.MultiVector(self.__distributed_map,y_data.T)
            exporter = Epetra.Export(self.__distributed_map,self.__operator_map)
            y[:] = 0
            y.Export(y_vec,exporter,Epetra.Add)
            return 0
        except Exception, e:
            print "Exception in LocalOperator.Apply:"
            print e
            return -1
            
def generate_maps(layers):
    """Generate the maps for the data distribution
    """
    
    # Need 2* number of dofs to convert complex to real
    ndofs = 2*sum([layers[i]['spaces']['ndof'] for i in range(len(layers))])
    distributed_map = Epetra.Map(-1,range(ndofs),0,comm)
    if comm.MyPID() == 0:
        num_my_elems = ndofs
    else:
        num_my_elems = 0
    operator_map = Epetra.Map(-1,num_my_elems,0,comm)

    return (operator_map,distributed_map)


def initialize_rhs(context,layers,evalFun,operator_map):
    """Generate the rhs vector"""

    pwc = layers[0]['spaces']['c']
    plc = layers[0]['spaces']['l']
    ndof = plc.globalDofCount()

    rhs_data = Epetra.MultiVector(operator_map,1)

    if comm.MyPID()==0:
        
        pwc = layers[0]['spaces']['c']
        plc = layers[0]['spaces']['l']

        fun = lib.createGridFunction(context,pwc,plc,evalFun)
        data = fun.projections()
        rhs_data[0,:2*ndof] = np.hstack([np.real(data),np.imag(data)])

    return rhs_data

def solve_system(operator,rhs,operator_map):
    """Solve the global system with GMRES"""

    from PyTrilinos import AztecOO
    x = Epetra.MultiVector(operator_map,1)
    solver = AztecOO.AztecOO(operator,x,rhs)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
    solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_none)
    solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_last)
    solver.Iterate(500,1E-4)
    return x

def save_output(context,layers,x):

    pass
        

if __name__=="__main__":
    
    mua = [.01, .02]
    mus = [1., .5]
    c = .3
    freq = 100e6
    omega = 2*np.pi*freq*1E-12
    alpha = 2.7439
    h=.5

    wavenumbers = compute_wavenumbers(mua,mus,c,omega)


    graph = {0:[1]}
    radii = [1.5, 2.5]
    layer_grids = [shapes.sphere(radii[i],h=h) for i in range(len(radii))]

    context = initialize_context()
    layers = initialize_layers(context,graph,layer_grids,wavenumbers)
    A = generate_local_operator(context,comm.MyPID(),graph,layers,alpha)
    
    op_map, dist_map = generate_maps(layers)
    local_op = LocalOperator(A,op_map,dist_map)

    def evalFun(point):
        return 1.

    rhs = initialize_rhs(context,layers,evalFun,op_map)

    x = solve_system(local_op,rhs,op_map)


    
