from PyTrilinos import Epetra
from bempp import lib,shapes

import numpy as np

test_vec = None


def blockmap(graph):
    ng = len(graph)
    nsurf = 0
    for k,v in graph.iteritems():
        nsurf = np.max((nsurf,np.max(graph[k])))
    nsurf = nsurf+1
    map = np.zeros((nsurf,nsurf),dtype='int32')
    # mark diagonal elements
    for i in range(nsurf):
        map[i,i] = 1
    # mark cross-surface blocks
    for i,v in graph.iteritems():
        if np.isscalar(v):
            map[i,v] = 1
            map[v,i] = 1
        else:
            nchild = len(v)
            for k in range(nchild):
                map[i,v[k]] = 1
                map[v[k],i] = 1
                # we also need to mark all siblings
                for s1 in range(nchild-1):
                    for s2 in range(nchild-s1-1):
                        si = v[s1];
                        sj = v[s1+s2+1];
                        map[si,sj] = 1
                        map[sj,si] = 1
    return map


# From a block layout and the number of processes, assign
# blocks to each processor and return the assignment as
# a matrix of processor indices

def procmap(map,nproc,full=False):
    nsurf = map.shape[0]
    pmap = np.zeros((nsurf,nsurf),dtype='int32')
    proc = 0
    # assign diagonal blocks
    for i in range(nsurf):
        pmap[i,i] = proc+1
        proc = (proc+1) % nproc
    # assign off-diagonal blocks in upper triangle
    for i in range(nsurf):
        for j in range(nsurf-i-1):
            if map[i,i+j+1] == 1:
                pmap[i,i+j+1] = proc+1
                proc = (proc+1) % nproc
    if full:
        # assign off-diagonal blocks in lower triangle
        for i in range(nsurf):
            for j in range(i):
                if map[i,j] == 1:
                    pmap[i,j] = proc+1
                    proc = (proc+1) % nproc
    return pmap


# From a processor map, return a dictionary with block
# assignments for each processor

def partition(pmap):
    nsurf = pmap.shape[0]
    nproc = np.max(pmap)
    sched = {}
    for i in range(nsurf):
        for j in range(nsurf):
            proc = pmap[i,j]
            if proc > 0:
                if proc in sched:
                    sched[proc].append([i,j])
                else:
                    sched[proc] = [[i,j]]
    return sched


def compute_wavenumbers_kappas(mu_absorption,mu_scattering,c,omega):
    """Compute a vector of complex wavenumbers from the vectors
       mu_absoprtion and mu_scattering.
    """

    n = len(mu_absorption)
    kappa = [1./(3.*(mu_absorption[i]+mu_scattering[i])) for i in range(n)]
    wavenumbers = [np.sqrt(mua[i]/kappa[i] + 1j*omega/(c*kappa[i])) for i in range(n)]

    return wavenumbers,kappa


def initialize_context():
    
    accuracyOptions = lib.createAccuracyOptions()
    accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
    accuracyOptions.doubleSingular.setRelativeQuadratureOrder(1)
    quadStrategy = lib.createNumericalQuadratureStrategy("float64","complex128",accuracyOptions)
    options = lib.createAssemblyOptions()
    options.setVerbosityLevel("low")
    aca_ops = lib.createAcaOptions()
    aca_ops.eps=1E-6
    options.switchToAcaMode(aca_ops)
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

def initialize_layers(context, graph, layer_grids, wavenumbers,kappas):
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
        layers[i]['kappa'] = kappas[i]
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
    kappa = layers[layer_id]['kappa']

    if layer_id == 0:
        I_00  = lib.createIdentityOperator(context,plc,pwc,plc,label="I00")
        I_01 = lib.createIdentityOperator(context,pwc,pwc,plc,label="I01")
        I_10 = lib.createIdentityOperator(context,plc,plc,pwc,label="I10")
        K = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,k,label="K10")
        S = 1./kappa*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,k,label="S11")

        return lib.createBlockedBoundaryOperator(context,[[I_00,2*alpha*I_01],[-.5*I_10-K,S]])
    else:
        kf = layers[layers[layer_id]['father']]['k']
        kappaf = layers[layers[layer_id]['father']]['kappa']
        DD = (-kappaf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc,pwc,plc,kf,label="Df"+str(layer_id)+"_"+str(layer_id))
              -kappa*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc,pwc,plc,k,label="D"+str(layer_id)+"_"+str(layer_id)))
        TT = (-lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc,pwc,plc,kf,label="Tf"+str(layer_id)+"_"+str(layer_id))
              -lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc,pwc,plc,k,label="T"+str(layer_id)+"_"+str(layer_id)))
        KK = (lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,kf,label="Kf"+str(layer_id)+"_"+str(layer_id))
              +lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc,plc,pwc,k,label="K"+str(layer_id)+"_"+str(layer_id)))
        SS = (-1./kappaf*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,kf,label="Sf"+str(layer_id)+"_"+str(layer_id))
              -1./kappa*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc,plc,pwc,k,label="S"+str(layer_id)+"_"+str(layer_id)))
        
        return lib.createBlockedBoundaryOperator(context,[[DD,TT],[KK,SS]])

def off_diagonal_block_father_son(context,layers,father_id,son_id):
    """Off-diagonal interaction between two different layers. Returns a tuple of two operators.
    """

    plc_father = layers[father_id]['spaces']['l']
    pwc_father = layers[father_id]['spaces']['c']
    plc_son = layers[son_id]['spaces']['l']
    pwc_son = layers[son_id]['spaces']['c']

    kf = layers[father_id]['k']
    kappaf = layers[father_id]['kappa']
    
    if father_id==0:
        null_00 = lib.createNullOperator(context,plc_son,pwc_father,plc_father,label="Null_00")
        null_01 = lib.createNullOperator(context,pwc_son,pwc_father,plc_father,label="Null_01")
        Kf = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_son,plc_father,pwc_father,kf,label="Kf0_"+str(son_id))
        Sf = 1./kappaf*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc_son,plc_father,pwc_father,kf,label="Sf0_"+str(son_id))

        op_father_son = lib.createBlockedBoundaryOperator(context,[[null_00,null_01],[Kf,-Sf]])

        Ds = kappaf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_father,pwc_son,plc_son,kf,label="Ds"+str(son_id)+"_0")
        Ts = lib.adjoint(Kf,pwc_son)
        Ks = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_father,plc_son,pwc_son,kf,label="Ks"+str(son_id)+"_0")
        Ss = lib.adjoint(Sf,plc_son)

        op_son_father = lib.createBlockedBoundaryOperator(context,[[Ds,Ts],[-Ks,Ss]])

    else:
        Df = kappaf*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_son,pwc_father,plc_father,kf)
        Tf = lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc_son,pwc_father,plc_father,kf)
        Kf = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_son,plc_father,pwc_father,kf)
        Sf = 1./kappaf*lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,pwc_son,plc_father,pwc_father,kf)

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
    kappa = layers[layers[id1]['father']]['kappa']

    D = kappa*lib.createModifiedHelmholtz3dHypersingularBoundaryOperator(context,plc_id2,pwc_id1,plc_id1,k)
    T = lib.createModifiedHelmholtz3dAdjointDoubleLayerBoundaryOperator(context,pwc_id2,pwc_id1,plc_id1,k)
    K = lib.createModifiedHelmholtz3dDoubleLayerBoundaryOperator(context,plc_id2,plc_id1,pwc_id1,k)
    S = 1./kappa*lib.createModifiedHelmholtz3dSingleLayerBoundaryOperator(context,pwc_id2,plc_id1,pwc_id1,k)

    D2 = lib.adjoint(D,pwc_id2)
    T2 = lib.adjoint(K,pwc_id2)
    K2  = lib.adjoint(T,plc_id2)
    S2  = lib.adjoint(S,plc_id2)

    op1 = lib.createBlockedBoundaryOperator(context,[[-D,-T],[K,-S]])
    op2 = lib.createBlockedBoundaryOperator(context,[[-D2,-T2],[K2,-S2]])
    
    return (op1,op2)


def generate_local_block_operator(context,my_proc,graph,layers,alpha):
    """Generate the local system matrix.
    """

    def copy_to_structure(brow,bcol,block,structure):
        """Small helper routine to copy a 2x2 block at the right position into a structure
        """
        structure.setBlock(2*brow,2*bcol,block.block(0,0))
        structure.setBlock(2*brow,2*bcol+1,block.block(0,1))
        structure.setBlock(2*brow+1,2*bcol,block.block(1,0))
        structure.setBlock(2*brow+1,2*bcol+1,block.block(1,1))

    nproc = Epetra.PyComm().NumProc()

    process_map = procmap(blockmap(graph),nproc)
    scheduler = partition(process_map)

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

    if not scheduler.has_key(my_proc+1):
        A = lib.createBlockedBoundaryOperator(context,structure)
        return A
        
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

class ProcessMaps(object):
    """Simple helper class to store maps"""

    def __init__(self,layers):
        self.__comm = Epetra.PyComm()
        self.__layers = layers
        self.__single_proc_map,self.__multiple_proc_map = self.generate_maps()
        self.__importer = Epetra.Import(self.__multiple_proc_map,self.__single_proc_map)
        self.__exporter = Epetra.Export(self.__multiple_proc_map,self.__single_proc_map)

    def generate_maps(self):
        """Generate the maps for the data distribution
        """
    
        # Need 2* number of dofs to convert complex to real
        layers = self.__layers
        ndofs = 2*sum([layers[i]['spaces']['ndof'] for i in range(len(layers))])
        multiple_proc_map = Epetra.Map(-1,range(ndofs),0,self.__comm)
        if self.__comm.MyPID() == 0:
            num_my_elems = ndofs
        else:
            num_my_elems = 0
        
        single_proc_map = Epetra.Map(-1,num_my_elems,0,self.__comm)

        return (single_proc_map,multiple_proc_map)

    @property
    def single_proc_map(self):
        return self.__single_proc_map

    @property
    def multiple_proc_map(self):
        return self.__multiple_proc_map

    @property
    def importer(self):
        return self.__importer

    @property
    def exporter(self):
        return self.__exporter

    @property
    def layers(self):
        return self.__layers

    
class TrilinosOperator(Epetra.Operator):
    """Define a Trilinos operator for the matvec product
    """

    def __init__(self,A,process_maps):
        Epetra.Operator.__init__(self)
        self.__label = "Operator"
        self.__A = A

        from bempp import tools
        self.__operator = tools.RealOperator(A.weakForm()) # Here the weak form is assembled
        self.__single_proc_map = process_maps.single_proc_map
        self.__multiple_proc_map = process_maps.multiple_proc_map
        self.__importer = process_maps.importer
        self.__exporter = process_maps.exporter
        self.__comm = Epetra.PyComm()
        self.__useTranspose = False

    def Label(self):
        return self.__label

    def OperatorDomainMap(self):
        return self.__single_proc_map

    def OperatorRangeMap(self):
        return self.__single_proc_map

    def Comm(self):
        return self.__comm

    def ApplyInverse(self,x,y):
        return -1

    def HasNormInf(self):
        return False

    def NormInf(self):
        return -1

    def SetUseTranspose(self, useTranspose):
        return -1

    def UseTranspose(self):
        return self.__useTranspose

    def Apply(self,x,y):
        try:
            # First import the vector into the distributed_map

            my_xvec = Epetra.MultiVector(self.__multiple_proc_map,x.NumVectors())
            my_xvec.Import(x,self.__importer,Epetra.Insert)

            # Now apply the local operator
            y_data = self.__operator.matmat(my_xvec.ExtractView().T)
 
            # Now export back

            y_vec = Epetra.MultiVector(self.__multiple_proc_map,y_data.T)
            y[:] = 0
            y.Export(y_vec,self.__exporter,Epetra.Add)
            return 0
        except Exception, e:
            print "Exception in LocalOperator.Apply:"
            print e
            return -1

class BlockDiagonalPreconditioner(Epetra.Operator):
    """Provide a block diagonal preconditioner"""
            
    def __init__(self,A,process_maps,context,aca_tol=1E-2):
        Epetra.Operator.__init__(self)
        self.__label = "DiagonalPreconditioner"
        self.__A = A
        self.__single_proc_map = process_maps.single_proc_map
        self.__multiple_proc_map = process_maps.multiple_proc_map
        self.__importer = process_maps.importer
        self.__exporter = process_maps.exporter
        self.__comm = Epetra.PyComm()
        self.__useTranspose = False
        self.__layers = process_maps.layers
        self.__context = context
        self.__blocks = self.buildBlockInverses(aca_tol)
        self.__ndofs_complex=sum([self.__layers[i]['spaces']['ndof'] for i in range(len(self.__layers))])

    def buildBlockInverses(self,aca_tol):
        """Generate the Block LU decompositions
        """

        from bempp import tools

        A = self.__A
        layers = self.__layers
        nproc = self.__comm.NumProc()
        my_id = self.__comm.MyPID()
        scheduler = partition(procmap(blockmap(graph),nproc))
        local_diag_blocks = []
        if scheduler.has_key(my_id+1):
            for elem in scheduler[my_id+1]:
                if elem[0] == elem[1]:
                    local_diag_blocks.append(elem[0])

        # Create a vector with index offsets
        offsets = np.cumsum([0]+[layer['spaces']['ndof'] for layer in layers])

        # Now create the LU blocks and store together with offsets
        blocks = []
        for id in local_diag_blocks:
            op00 = A.block(2*id,2*id)
            op01 = A.block(2*id,2*id+1)
            op10 = A.block(2*id+1,2*id)
            op11 = A.block(2*id+1,2*id+1)
            op = lib.createBlockedBoundaryOperator(self.__context,[[op00,op01],[op10,op11]])
            print "Generate approximate LU for block %(id)i on process %(my_id)i" % {'id':id, 'my_id':my_id}
            lu = lib.acaOperatorApproximateLuInverse(op.weakForm().asDiscreteAcaBoundaryOperator(),aca_tol)
            print "Finished generating approximate LU for block %(id)i on process %(my_id)i" % {'id':id, 'my_id':my_id}
            blocks.append({'lu':tools.RealOperator(lu),
                           'start':offsets[id],
                           'end':offsets[id+1]
                           })
        return blocks    
        
    def Label(self):
        return self.__label

    def OperatorDomainMap(self):
        return self.__single_proc_map

    def OperatorRangeMap(self):
        return self.__single_proc_map

    def Comm(self):
        return self.__comm

    def ApplyInverse(self,x,y):
        
        from bempp import tools

        n = self.__ndofs_complex
        try:
            # First import the vector into the distributed_map

            my_xvec = Epetra.MultiVector(self.__multiple_proc_map,x.NumVectors())
            my_xvec.Import(x,self.__importer,Epetra.Insert)
            
            res = np.zeros_like(my_xvec.ExtractView().T)
            # Now apply the local operators
            for block in self.__blocks:
                # Extract the correct vector
                v = my_xvec.ExtractView().T
                s = block['start']
                e = block['end']
                tmp = np.vstack([v[s:e,:],v[n+s:n+e,:]])
                #Multiply
                tmp_res = block['lu'].matmat(tmp)
                # Now distribute back
                res[s:e,:] = tmp_res[:e-s,:]
                res[n+s:n+e,:] = tmp_res[e-s:,:]
 
            # Now export back

            y_vec = Epetra.MultiVector(self.__multiple_proc_map,res.T)
            y[:] = 0
            y.Export(y_vec,self.__exporter,Epetra.Add)
            return 0
        except Exception, e:
            print "Exception in BlockDiagonalPreconditioner.ApplyInverse:"
            print e
            return -1
        

    def HasNormInf(self):
        return False

    def NormInf(self):
        return -1

    def SetUseTranspose(self, useTranspose):
        return -1

    def UseTranspose(self):
        return self.__useTranspose

    def Apply(self,x,y):
        return -1
    
    def Blocks(self):
        return self.__blocks

def initialize_rhs(context,layers,evalFun,process_maps):
    """Generate the rhs vector"""

    pwc = layers[0]['spaces']['c']
    plc = layers[0]['spaces']['l']
    ndof = plc.globalDofCount()
    total_dofs = sum([layers[i]['spaces']['ndof'] for i in range(len(layers))])

    rhs_data = Epetra.MultiVector(process_maps.single_proc_map,1)

    if Epetra.PyComm().MyPID()==0:
        
        pwc = layers[0]['spaces']['c']
        plc = layers[0]['spaces']['l']

        fun = lib.createGridFunction(context,pwc,plc,evalFun)
        data = fun.projections()
        rhs_data[0,:ndof] = np.real(data)
        rhs_data[0,total_dofs:total_dofs+ndof]=np.imag(data)

    return rhs_data

def solve_system(operator,rhs,process_maps,tol=1E-5,prec=None):
    """Solve the global system with GMRES"""

    from PyTrilinos import AztecOO
    x = Epetra.MultiVector(process_maps.single_proc_map,1)
    solver = AztecOO.AztecOO(operator,x,rhs)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
    if prec is None:
        solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_none)
    else:
        solver.SetPrecOperator(P)
    solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_last)
    solver.Iterate(500,tol)
    return x

def save_output(context,layers,x):
    """Create vtk files from the result"""

    ndofc = sum([layers[i]['spaces']['ndof'] for i in range(len(layers))])
    tmp = x.ExtractView().T
    xc = tmp[:ndofc,:]+1j*tmp[ndofc:,:]

    if Epetra.PyComm().MyPID()==0:
        n = 0
        for i in range(len(layers)):
            plc = layers[i]['spaces']['l']
            pwc = layers[i]['spaces']['c']
            nplc = plc.globalDofCount()
            npwc = pwc.globalDofCount()
            ndof = nplc+npwc

            coeff_dirichlet = xc[n:n+nplc,0]
            coeff_neumann = xc[n+nplc:n+nplc+npwc,0]
            n = n+ndof

            ufun = lib.createGridFunction(context,plc,pwc,coefficients=coeff_dirichlet)
            vfun = lib.createGridFunction(context,pwc,plc,coefficients=coeff_neumann)

            ufun.exportToVtk("vertex_data","dirichlet_data","u"+str(i))
            vfun.exportToVtk("cell_data","neumann_data","v"+str(i))
            

if __name__=="__main__":
    
    mua = [.01, .02, .03]
    mus = [1., .5, .4]
    c = .3
    freq = 100e6
    omega = 2*np.pi*freq*1E-12
    alpha = 2.7439
    h=.1

    wavenumbers,kappas = compute_wavenumbers_kappas(mua,mus,c,omega)

    graph = {0:[1,2]}
    radii = [1., .2, .2]
    origins = [(0,0,0),(0,-.3,0),(0,.3,0.)]
    layer_grids = [shapes.sphere(radius=radii[i],origin=origins[i],h=h) for i in range(len(radii))]

    context = initialize_context()
    layers = initialize_layers(context,graph,layer_grids,wavenumbers,kappas)
    process_maps = ProcessMaps(layers)
    A = generate_local_block_operator(context,Epetra.PyComm().MyPID(),graph,layers,alpha)
    P = BlockDiagonalPreconditioner(A,process_maps,context)

    operator = TrilinosOperator(A,process_maps)

    def evalFun(point):
        return 1.

    rhs = initialize_rhs(context,layers,evalFun,process_maps)

    x = solve_system(operator,rhs,process_maps,tol=1E-10,prec=P)
    save_output(context,layers,x)

    
