import numpy as np
import ctypes as ct
import cgen

from pdb import set_trace
from sympy import Mod
from functools import reduce
from collections import OrderedDict

from devito.ir.iet.utils import array_alloc_check, get_first_space_dim_index
from devito.tools import timed_pass
from devito.symbolics import (CondNe, Macro, String)

from devito.types import (CustomDimension, Array, Symbol, TimeDimension, off_t, zfp_type)
from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Call, Conditional, 
                           FindNodes, Transformer)
from devito.ir.equations import IREq, ClusterizedEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward, PARALLEL, AFFINE)

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree, **kwargs):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """

    ooc = kwargs['options']['out-of-core']
    if ooc.function.save: raise ValueError("Out of core incompatible with TimeFunction save functionality")
    is_mpi = kwargs['options']['mpi']
    time_iterator = None
    
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            iet_body = queues.pop(i)
            if(ooc):
                iet_body = _ooc_build(iet_body, kwargs['sregistry'].nthreads, ooc, is_mpi, time_iterator)
                return List(body=iet_body)
            else:                
                return List(body=iet_body)

        elif i.is_Exprs:
            exprs = []
            for e in i.exprs:
                if e.is_Increment:
                    exprs.append(Increment(e))
                else:
                    exprs.append(Expression(e, operation=e.operation))
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            iteration_nodes = queues.pop(i)
            if isinstance(i.dim, TimeDimension) and ooc and ooc.mode == 'forward':
                iteration_nodes.append(Section("write_temp"))
                time_iterator = i.sub_iterators[0]
            elif isinstance(i.dim, TimeDimension) and ooc and ooc.mode == 'gradient':
                iteration_nodes.insert(0, Section("read_temp"))
                time_iterator = i.sub_iterators[0]

            body = Iteration(iteration_nodes, i.dim, i.limits, direction=i.direction,
                             properties=i.properties, uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(queues.pop(i), i.halo_scheme)

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i, None))

        queues.setdefault(i.parent, []).append(body)

    assert False


@timed_pass(name='ooc_build')
def _ooc_build(iet_body, nthreads, ooc, is_mpi, time_iterator):
    """
    This private method builds a iet_body (list) with out-of-core nodes.

    Args:
        iet_body (List): a list of nodes
        nthreads (NThreads): symbol representing nthreads parameter of OpenMP
        func (Function): I/O TimeFunction
        out_of_core (string): 'forward' or 'gradient'
        is_mpi (bool): MPI execution flag

    Returns:
        List : iet_body is a list of nodes
    """
    func = ooc.function
    out_of_core = ooc.mode
    is_compression = ooc.compression
    is_forward = out_of_core == 'forward'

    # Creates nthreads once again in order to enable the ignoreDefinition flag
    #nthreads = NThreads(ignoreDefinition=True)

    ######## Dimension and symbol for iteration spaces ########
    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads)    
    iSymbol = Symbol(name="i", dtype=np.int32)
    
    # testScalar = Scalar(name="testScalar", ignoreDefinition=True)


    ######## Build files and counters arrays ########
    filesArray = Array(name='files', dimensions=[nthreadsDim], dtype=np.int32)
    countersArray = Array(name='counters', dimensions=[nthreadsDim], dtype=np.int32)
    metasArray = Array(name='metas', dimensions=[nthreadsDim], dtype=np.int32)
    sptArray = Array(name='spt', dimensions=[nthreadsDim], dtype=np.int32)


    ######## Build open section ########
    # TODO: why metas doesn't appear as input when we print the hole operator?
    openSection = open_build(filesArray, countersArray, metasArray, nthreadsDim, nthreads, is_forward, iSymbol, is_compression)

    ######## Build func_size var ########
    func_size = Symbol(name=func.name+"_size", dtype=np.uint64) 
    funcSizeExp, floatSizeInit = func_size_build(func, func_size)

    if is_compression: 
        ######## Build compress/decompress section ########
        compress_or_decompress_build(iet_body, iSymbol, is_forward, func, nthreads) 
    else:
        ######## Build write/read section ########    
        write_or_read_build(iet_body, is_forward, nthreads, filesArray, iSymbol, func_size, func, time_iterator, countersArray, is_mpi)
        
        ######## Build write_size var ########
        size_name = 'write_size' if is_forward else 'read_size'
        ioSize = Symbol(name=size_name, dtype=np.int64)
        ioSizeExp = io_size_build(ioSize, func_size, func)
    
    ######## Build close section ########
    closeSection = close_build(nthreads, filesArray, iSymbol, nthreadsDim)    

    ######## Build save call ########
    # timerProfiler = Timer(profiler.name, [], ignoreDefinition=True)
    # saveCall = Call(name='save_temp', arguments=[nthreads, timerProfiler, ioSize])        
    
    #TODO: Generate blank lines between sections
    iet_body.insert(0, funcSizeExp)
    iet_body.insert(0, floatSizeInit)
    iet_body.insert(0, openSection)
    iet_body.append(closeSection)
    iet_body.append(ioSizeExp)
    # iet_body.append(saveCall)

    return iet_body


def compress_or_decompress_build(iet_body, iSymbol, is_forward, funcStencil, nthreads):
    
    uVecSize1 = funcStencil.symbolic_shape[1]
    uSizeDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(uSizeDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)    
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    
    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    
    if is_forward:
        section = compress_build(funcStencil, iSymbol, pragma, uSizeDim, tid, cTidEq, ispace)
    else:
        section = decompress_build(funcStencil, iSymbol, pragma, uSizeDim, tid, cTidEq, ispace)
    

def compress_build(funcStencil, iSymbol,pragma, uSizeDim, tid, cTidEq, ispace):
    
    uVecSize1 = funcStencil.symbolic_shape[1]    
    itNodes=[]
    
    field = Symbol(name='field', dtype=zfp_type)

    itNodes.append(Expression(cTidEq, None, True))
    
    compressIteration = Iteration(itNodes, uSizeDim, uVecSize1-1, pragmas=[pragma])
    
    return Section("compress", compressIteration)

def decompress_build(funcStencil, iSymbol, pragma, uSizeDim, tid, cTidEq, ispace):
    
    uVecSize1 = funcStencil.symbolic_shape[1]    
    itNodes=[]
    
    itNodes.append(Expression(cTidEq, None, True))
    
    decompressIteration = Iteration(itNodes, uSizeDim, uVecSize1-1, direction=Backward, pragmas=[pragma])
    
    return Section("decompress", decompressIteration)




def write_or_read_build(iet_body, is_forward, nthreads, filesArray, iSymbol, func_size, funcStencil, t0, countersArray, is_mpi):
    """
    Builds the read or write section of the operator, depending on the out_of_core mode.
    Replaces the temporary section at the end of the time iteration by the read or write section.   

    Args:
        iet_body (List): list of IET nodes 
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        countersArray (array): pointer of allocated memory of nthreads dimension. Each place has a size of int

    """
    
    uVecSize1 = funcStencil.symbolic_shape[1]
    if is_forward:
        ooc_section = write_build(nthreads, filesArray, iSymbol, func_size, funcStencil, uVecSize1, t0, is_mpi)
        temp_name = 'write_temp'
    else: # gradient
        ooc_section = read_build(nthreads, filesArray, iSymbol, func_size, funcStencil, uVecSize1, t0, countersArray)
        temp_name = 'read_temp'  

    sections = FindNodes(Section).visit(iet_body)
    temp_sec = next((section for section in sections if section.name == temp_name), None)
    mapper={temp_sec: ooc_section}

    timeIndex = next((i for i, node in enumerate(iet_body) if isinstance(node, Iteration) and isinstance(node.dim, TimeDimension)), None)
    transformedIet = Transformer(mapper).visit(iet_body[timeIndex])
    iet_body[timeIndex] = transformedIet

def write_build(nthreads, filesArray, iSymbol, func_size, funcStencil, uVecSize1, t0, is_mpi):
    """
    This method inteds to code gradient.c write section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        uVecSize1 (FieldFromPointer): size of a vector u

    Returns:
        Section: complete wrie section
    """
    
    
    uSizeDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(uSizeDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)
    itNodes = []

    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    itNodes.append(Expression(cTidEq, None, True))
    
    ret = Symbol(name="ret", dtype=np.int32)
    writeCall = Call(name="write", arguments=[filesArray[tid], funcStencil[t0, iSymbol], func_size], retobj=ret)
    itNodes.append(writeCall)
    
    pstring = String("\"Write size mismatch with function slice size\"")

    condNodes = [Call(name="perror", arguments=pstring)]
    condNodes.append(Call(name="exit", arguments=1))
    cond = Conditional(CondNe(ret, func_size), condNodes)
    itNodes.append(cond)

    # TODO: Pragmas should depend on the user's selected optimization options and be generated by the compiler
    if is_mpi:
        pragma = cgen.Pragma("omp parallel for schedule(static,1)")
    else:
        pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    writeIteration = Iteration(itNodes, uSizeDim, uVecSize1-1, pragmas=[pragma])

    return Section("write", writeIteration)

def read_build(nthreads, filesArray, iSymbol, func_size, funcStencil, uVecSize1, t0, counters):
    """
    This method inteds to code gradient.c read section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        func_size (Symbol): the funcStencil size
        funcStencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        uVecSize1 (FieldFromPointer): size of a vector u
        counters (array): pointer of allocated memory of nthreads dimension. Each place has a size of int

    Returns:
        section (Section): complete read section
    """
    
    #  pragma omp parallel for schedule(static,1) num_threads(nthreads)
    #  0 <= i <= u_vec->size[1]-1
    #  TODO: Pragmas should depend on user's selected optimization options and generated by the compiler
    uVecSize1 = funcStencil.symbolic_shape[1]
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    iDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(iDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)
    itNodes = []

    # int tid = i%nthreads;
    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    itNodes.append(Expression(cTidEq, None, True))
    
    # off_t offset = counters[tid] * func_size;
    # lseek(files[tid], -1 * offset, SEEK_END);
    # TODO: make offset be a off_t
    offset = Symbol(name="offset", dtype=off_t)
    offsetEq = IREq(offset, (-1)*counters[tid]*func_size)
    cOffsetEq = ClusterizedEq(offsetEq, ispace=ispace)
    itNodes.append(Expression(cOffsetEq, None, True))    
    itNodes.append(Call(name="lseek", arguments=[filesArray[tid], offset, Macro("SEEK_END")]))

    # int ret = read(files[tid], u[t0][i], func_size);
    ret = Symbol(name="ret", dtype=np.int32)
    readCall = Call(name="read", arguments=[filesArray[tid], funcStencil[t0, iSymbol], func_size], retobj=ret)
    itNodes.append(readCall)

    # printf("%d", ret);
    # perror("Cannot open output file");
    # exit(1);
    pret = String("'%d', ret")
    pstring = String("\"Cannot open output file\"")
    condNodes = [
        Call(name="printf", arguments=pret),
        Call(name="perror", arguments=pstring), 
        Call(name="exit", arguments=1)
    ]
    cond = Conditional(CondNe(ret, func_size), condNodes) # if (ret != func_size)
    itNodes.append(cond)
    
    # counters[tid] = counters[tid] + 1
    newCountersEq = IREq(counters[tid], 1)
    cNewCountersEq = ClusterizedEq(newCountersEq, ispace=ispace)
    itNodes.append(Increment(cNewCountersEq))
        
    readIteration = Iteration(itNodes, iDim, uVecSize1-1, direction=Backward, pragmas=[pragma])
    
    section = Section("read", readIteration)

    return section

def open_build(filesArray, countersArray, metasArray, nthreadsDim, nthreads, is_forward, iSymbol, is_compression):
    """
    This method inteds to code open section for both Forward and Gradient operators.
    
    Args:
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        countersArray (counters): pointer of allocated memory of nthreads dimension. Each place has a size of int
        metasArray (Array): some array
        nthreadsDim (CustomDimension): dimension from 0 to nthreads 
        nthreads (NThreads): number of threads
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        is_compression (bool): True for the use of compression; False otherwise

    Returns:
        Section: open section
    """
    
    # Test files array and exit if get wrong
    filesArrCond = array_alloc_check(filesArray) #  Forward
    
    #Call open_thread_files
    funcArgs = [filesArray, nthreads]
    if is_compression:
        funcArgs = [filesArray, metasArray, nthreads]
    open_thread_call = Call(name='open_thread_files', arguments=funcArgs)

    # Open section body
    body = [filesArrCond, open_thread_call]
    
    if not is_forward and not is_compression:
        countersArrCond = array_alloc_check(countersArray) # gradient
        body.append(countersArrCond)
        
        intervalGroup = IntervalGroup((Interval(nthreadsDim, 0, nthreads)))
        cNewCountersEq = ClusterizedEq(IREq(countersArray[iSymbol], 1), ispace=IterationSpace(intervalGroup))
        openIterationGrad = Iteration(Expression(cNewCountersEq, None, False), nthreadsDim, nthreads-1)
        body.append(openIterationGrad)
        
    return Section("open", body)

def close_build(nthreads, filesArray, iSymbol, nthreadsDim):
    """
    This method inteds to code gradient.c close section.
    Obs: maybe the desciption of the variables should be better

    Args:
        nthreads (NThreads): symbol of number of threads
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        iSymbol (Symbol): symbol of the iterator index i
        nthreadsDim (CustomDimension): dimension from 0 to nthreads

    Returns:
        section (Section): complete close section
    """

    # close(files[i]);
    itNode = Call(name="close", arguments=[filesArray[iSymbol]])    
    
    # for(int i=0; i < nthreads; i++) --> for(int i=0; i <= nthreads-1; i+=1)
    closeIteration = Iteration(itNode, nthreadsDim, nthreads-1)
    
    section = Section("close", closeIteration)
    
    return section

def func_size_build(funcStencil, func_size):
    """
    Generates float_size init call and the init function size expression.

    Args:
        funcStencil (AbstractFunction): I/O function
        func_size (Symbol): Symbol representing the I/O function size

    Returns:
        funcSizeExp: Expression initializing the function size
        floatSizeInit: Call initializing float_size
    """

    floatSize = Symbol(name="float_size", dtype=np.uint64)
    floatString = String(r"float")
    floatSizeInit = Call(name="sizeof", arguments=[floatString], retobj=floatSize)
    
    # TODO: Function name must come from user?
    sizes = funcStencil.symbolic_shape[2:]
    funcEq = IREq(func_size, (reduce(lambda x, y: x * y, sizes) * floatSize))
    funcSizeExp = Expression(ClusterizedEq(funcEq, ispace=None), None, True)

    return funcSizeExp, floatSizeInit

def io_size_build(ioSize, func_size, funcStencil):
    """
    Generates init expression calculating io_size.

    Args:
        ioSize (Symbol): Symbol representing the total amount of I/O data
        func_size (Symbol): Symbol representing the I/O function size
        funcStencil (Function): function signature (Ex.: func(t, x, y, z))

    Returns:
        funcSizeExp: Expression initializing ioSize
    """

    time_M = funcStencil.time_dim.symbolic_max
    time_m = funcStencil.time_dim.symbolic_min
    
    first_space_dim_index = get_first_space_dim_index(funcStencil.dimensions)
    
    #TODO: Field and pointer must be retrieved from somewhere
    # funcSize1 = FieldFromPointer(f"size[{first_space_dim_index}]", funcStencil._C_name)
    funcSize1 = funcStencil.symbolic_shape[first_space_dim_index]
    
    ioSizeEq = IREq(ioSize, ((time_M - time_m+1) * funcSize1 * func_size))

    return Expression(ClusterizedEq(ioSizeEq, ispace=None), None, True)
