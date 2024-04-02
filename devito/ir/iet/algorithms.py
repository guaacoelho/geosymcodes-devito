import numpy as np
import ctypes as ct
import cgen

from pdb import set_trace
from sympy import Mod, Or, Not
from functools import reduce
from collections import OrderedDict

from devito.ir.iet.utils import array_alloc_check, update_iet, get_compress_mode_function
from devito.tools import timed_pass
from devito.symbolics import (CondNe, Macro, String, Null, Byref, SizeOf)

from devito.types import (CustomDimension, Array, Symbol, Pointer, TimeDimension, PointerArray,
                          NThreads, off_t, zfp_type, size_t, zfp_field, bitstream, zfp_stream, size_t)
from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Call, Conditional)
from devito.ir.equations import IREq, ClusterizedEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward, PARALLEL, AFFINE)

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree, **kwargs):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """
    ooc = kwargs['options']['out-of-core']
    time_iterators = None

    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            iet_body = queues.pop(i)
            if(ooc):
                iet_body = _ooc_build(iet_body, ooc, kwargs['sregistry'].nthreads, kwargs['options']['mpi'], kwargs['language'], time_iterators)               
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
                if ooc.compression:
                    iteration_nodes.append(Section("compress_temp"))
                else:
                    iteration_nodes.append(Section("write_temp"))
                time_iterators = i.sub_iterators
            elif isinstance(i.dim, TimeDimension) and ooc and ooc.mode == 'gradient':
                if ooc.compression:
                    # TODO: Move decompress section to the top (idx 0) and test
                    iteration_nodes.append(Section("decompress_temp"))
                else:
                    iteration_nodes.insert(0, Section("read_temp"))
                time_iterators = i.sub_iterators

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
def _ooc_build(iet_body, ooc, nt, is_mpi, language, time_iterators):
    """
    This private method builds a iet_body (list) with out-of-core nodes.

    Args:
        iet_body (List): a list of nodes
        ooc (Object): out of core parameters
        nt (NThreads): symbol representing nthreads parameter of OpenMP
        is_mpi (bool): MPI execution flag
        language (str): language set for the operator (C, openmp or openacc)
        time_iterators(Dimension): iterator used as index in each timestep

    Returns:
        List : iet_body is a list of nodes
    """
    funcs = ooc.functions
    out_of_core = ooc.mode
    ooc_compression = ooc.compression    

    if language != 'openmp':
       raise ValueError("Out of core requires OpenMP. Language parameter must be openmp, got %s" % language)
    
    for func in funcs:
        if func.save:
            raise ValueError("Out of core incompatible with TimeFunction save functionality on %s" % func.name)
    
    if ooc_compression and len(funcs) > 1:
        raise ValueError("Multi Function currently does not support compression")
    
    if is_mpi and len(funcs) > 1:
        raise ValueError("Multi Function currently does not support multi process")
    
    funcs_dict = dict((func.name, func) for func in funcs)
    is_forward = out_of_core == 'forward'
    time_iterator = time_iterators[0]

    ######## Dimension and symbol for iteration spaces ########
    nthreads = nt or NThreads()
    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads)    
    iSymbol = Symbol(name="i", dtype=np.int32)
    

    ######## Build files and counters arrays ########
    files_dict = dict()
    counters_dict = dict()
    for func in funcs:
        filesArray = Array(name=func.name + '_files', dimensions=[nthreadsDim], dtype=np.int32)
        countersArray = Array(name=func.name + '_counters', dimensions=[nthreadsDim], dtype=np.int32)
        files_dict.update({func.name: filesArray})
        counters_dict.update({func.name: countersArray})

    # Compression arrays
    metasArray = Array(name='metas', dimensions=[nthreadsDim], dtype=np.int32)
    sptArray = Array(name='spt', dimensions=[nthreadsDim], dtype=np.int32)
    offsetArray = Array(name='offset', dimensions=[nthreadsDim], dtype=off_t)
    slices_size = PointerArray(name='slices_size', dimensions=[nthreadsDim], array=Array(name='slices_size', dimensions=[nthreadsDim], dtype=size_t, ignoreDefinition=True))


    ######## Build open section ########
    openSection = open_build(files_dict, counters_dict, metasArray, sptArray, offsetArray, nthreadsDim, nthreads, is_forward, iSymbol, ooc_compression, slices_size)

    ######## Build func_size var ########
    floatSize = Symbol(name="float_size", dtype=np.uint64)
    floatSizeInit = Call(name="sizeof", arguments=[String(r"float")], retobj=floatSize)

    func_sizes_dict = {}
    func_sizes_symb_dict={}
    for func in funcs:
        func_size = Symbol(name=func.name+"_size", dtype=np.uint64) 
        funcSizeExp = func_size_build(func, func_size, floatSize)
        func_sizes_dict.update({func.name: funcSizeExp})
        func_sizes_symb_dict.update({func.name: func_size})

    if ooc_compression:                     
        ######## Build compress/decompress section ########
        compress_or_decompress_build(files_dict, metasArray, iet_body, iSymbol, is_forward, funcs_dict, nthreadsDim, nthreads, time_iterators, sptArray, slices_size, offsetArray, ooc_compression) 
    else:
        ######## Build write/read section ########    
        write_or_read_build(iet_body, is_forward, nthreads, files_dict, iSymbol, func_sizes_symb_dict, funcs_dict, time_iterator, counters_dict, is_mpi)
    
    
    ######## Build close section ########
    closeSection = close_build(nthreads, files_dict, iSymbol, nthreadsDim)
    
    #TODO: Generate blank lines between sections
    for size_init in func_sizes_dict.values():
        iet_body.insert(0, size_init)
    iet_body.insert(0, floatSizeInit)
    iet_body.insert(0, openSection)
    iet_body.append(closeSection)
    
    ### Free slices memory
    if ooc_compression:
        closeSlices = closeSlices_build(nthreads, iSymbol, slices_size)
        iet_body.append(closeSlices)
    
    return iet_body

def closeSlices_build(nthreads, iSymbol, slices_size):
    """
    This method inteds to ls gradient.c free slices_size array memory.
    Obs: code creates variables that already exists on the previous code 

    Args:
        nthreads (NThreads): symbol of number of threads

    Returns:
        closeIteration (Iteration): complete iteration loop 
    """

    # tid dimension
    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads)

    # free(slices_size[i]);
    itNode = Call(name="free", arguments=[slices_size[iSymbol]])    
    
    # for(int tid=0; tid < nthreads; i++) --> for(int tid=0; tid <= nthreads-1; tid+=1)
    closeIteration = Iteration(itNode, nthreadsDim, nthreads-1)
    
    return closeIteration

def compress_or_decompress_build(files_dict, metasArray, iet_body, iSymbol, is_forward, funcs_dict, nthreadsDim, nthreads, time_iterators, sptArray, slices_size, offsetArray, ooc_compression):
    """
    This function decides if it is either a compression or a decompression

    Args:
        filesArray (Array): array of files
        metasArray (Array): array of metadata
        iet_body (List): IET body nodes
        iSymbol (Symbol): iterator symbol
        is_forward (bool): if True, it is forward. It is gradient otherwise
        funcStencil (Function): Function defined by user for Operator
        nthreads (NThreads): number of threads
        time_iterator (tuple): time iterator indexes
        sptArray (Array): array of slices per thread
        offsetArray (Array): array of offset
        slices_size (PointerArray): 2d-array of slices
        ooc_compression (CompressionConfig): object with compression settings
    """
    
    # TODO: Temporary workaround, while compression mode supports only one Function
    funcStencil = next(iter(funcs_dict.values()))
    filesArray = next(iter(files_dict.values()))

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
        ooc_section = compress_build(filesArray, metasArray, funcStencil, iSymbol, pragma, uSizeDim, tid, cTidEq, ispace, time_iterators[0], ooc_compression)
        temp_name = "compress_temp"
    else:
        ooc_section = decompress_build(filesArray, funcStencil, iSymbol, pragma, nthreadsDim, uSizeDim, tid, cTidEq, ispace, time_iterators[-1], sptArray, slices_size, offsetArray, ooc_compression)
        temp_name = "decompress_temp"

    update_iet(iet_body, temp_name, ooc_section)      
    

def compress_build(filesArray, metasArray, funcStencil, iSymbol, pragma, uSizeDim, tid, cTidEq, ispace, t0, ooc_compression):
    """
    This function generates compress section.

    Args:
        filesArray (Array): array of files
        metasArray (Array): array of metadata
        funcStencil (Function): Function defined by user for Operator
        iSymbol (Symbol): iterator symbol
        pragma (Pragma): omp pragma directives
        uSizeDim (CustomDimension): symbolic dimension of loop
        tid (Symbol): iterator index symbol
        cTidEq (ClusterizedEq): expression that defines tid --> int tid = i%nthreads
        ispace (IterationSpace): space of iteration
        t0 (ModuloDimension): time iterator index for compression
        ooc_compression (CompressionConfig): object with compression settings

    Returns:
        Section: compress section
    """
    
    uVecSize1 = funcStencil.symbolic_shape[1]    
    itNodes=[]
    ifNodes=[]
    
    Type = Symbol(name='type', dtype=zfp_type)
    TypeEq = IREq(Type, String(r"zfp_type_float"))
    cTypeEq = ClusterizedEq(TypeEq, ispace=ispace)
    itNodes.append(Expression(cTypeEq, None, True))
    
    itNodes.append(Expression(cTidEq, None, True))
    
    field = Pointer(name="field", dtype=ct.POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=ct.POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=size_t)
    buffer = Pointer(name="buffer", dtype=ct.c_void_p)
    stream = Pointer(name="stream", dtype=ct.POINTER(bitstream))
    zfpsize = Symbol(name="zfpsize", dtype=size_t)
    
    itNodes.append(Call(name="zfp_field_2d", arguments=[funcStencil[t0,iSymbol], Type, funcStencil.symbolic_shape[2], funcStencil.symbolic_shape[3]], retobj=field))
    itNodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    itNodes.append(get_compress_mode_function(ooc_compression, zfp, field, Type))
    itNodes.append(Call(name="zfp_stream_maximum_size", arguments=[zfp, field], retobj=bufsize))
    itNodes.append(Call(name="malloc", arguments=[bufsize], retobj=buffer))
    itNodes.append(Call(name="stream_open", arguments=[bufsize, bufsize], retobj=stream))
    itNodes.append(Call(name="zfp_stream_set_bit_stream", arguments=[zfp, stream]))
    itNodes.append(Call(name="zfp_stream_rewind", arguments=[zfp]))
    itNodes.append(Call(name="zfp_compress", arguments=[zfp, field], retobj=zfpsize))
    
    ifNodes.append(Call(name="fprintf", arguments=[String(r"stderr"), String("\"compression failed\\n\"")]))
    ifNodes.append(Call(name="exit", arguments=1))
    itNodes.append(Conditional(Not(zfpsize), ifNodes))
    
    itNodes.append(Call(name="write", arguments=[filesArray[tid], buffer, zfpsize]))
    itNodes.append(Call(name="write", arguments=[metasArray[tid], Byref(zfpsize), SizeOf(String(r"size_t"))]))
    
    itNodes.append(Call(name="zfp_field_free", arguments=[field]))
    itNodes.append(Call(name="zfp_stream_close", arguments=[zfp]))
    itNodes.append(Call(name="stream_close", arguments=[stream]))
    itNodes.append(Call(name="free", arguments=[buffer]))
    
    compressSection = [Expression(cTypeEq, None, True), Iteration(itNodes, uSizeDim, uVecSize1-1, pragmas=[pragma])]
    return Section("compress", compressSection)

def decompress_build(filesArray, funcStencil, iSymbol, pragma, nthreadsDim, uSizeDim, tid, cTidEq, ispace, t2, sptArray, slices_size, offsetArray, ooc_compression):
    """
    This function generates decompress section.

    Args:
        filesArray (Array): array of files
        funcStencil (Function): Function defined by user for Operator
        iSymbol (Symbol): iterator symbol
        pragma (Pragma): omp pragma directives
        uSizeDim (CustomDimension): symbolic dimension of loop
        tid (Symbol): iterator index symbol
        cTidEq (ClusterizedEq): expression that defines tid --> int tid = i%nthreads
        ispace (IterationSpace): space of iteration
        t2 (ModuloDimension): time iterator index for compression
        sptArray (Array): array of slices per thread
        slices_size (PointerArray): 2d-array of slices
        offsetArray (Array): array of offset
        ooc_compression (CompressionConfig): object with compression settings

    Returns:
        Section: decompress section
    """
    
    uVecSize1 = funcStencil.symbolic_shape[1]    
    itNodes=[]
    if1Nodes=[]
    if2Nodes=[]
    
    itNodes.append(Expression(cTidEq, None, True))
    
    Type = Symbol(name='type', dtype=zfp_type)
    field = Pointer(name="field", dtype=ct.POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=ct.POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=off_t)
    buffer = Pointer(name="buffer", dtype=ct.c_void_p)
    stream = Pointer(name="stream", dtype=ct.POINTER(bitstream))
    Slice = Symbol(name="slice", dtype=np.int32)
    ret = Symbol(name="ret", dtype=np.int32)
    
    # TypeEq = IREq(Type, Symbol(name="zfp_type_float", dtype=ct.c_int))
    TypeEq = IREq(Type, String(r"zfp_type_float"))
    cTypeEq = ClusterizedEq(TypeEq, ispace=ispace)
    itNodes.append(Expression(cTypeEq, None, True))
    
    itNodes.append(Call(name="zfp_field_2d", arguments=[funcStencil[t2,iSymbol], Type, funcStencil.symbolic_shape[2], funcStencil.symbolic_shape[3]], retobj=field))
    itNodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    itNodes.append(get_compress_mode_function(ooc_compression, zfp, field, Type))
    itNodes.append(Call(name="zfp_stream_maximum_size", arguments=[zfp, field], retobj=bufsize))
    itNodes.append(Call(name="malloc", arguments=[bufsize], retobj=buffer))
    itNodes.append(Call(name="stream_open", arguments=[bufsize, bufsize], retobj=stream))
    itNodes.append(Call(name="zfp_stream_set_bit_stream", arguments=[zfp, stream]))
    itNodes.append(Call(name="zfp_stream_rewind", arguments=[zfp]))
    
    SliceEq = IREq(Slice, sptArray[tid])
    cSliceEq = ClusterizedEq(SliceEq, ispace=ispace)
    itNodes.append(Expression(cSliceEq, None, True))
    
    offsetIncr = IREq(offsetArray[tid], slices_size[tid, Slice])
    cOffsetIncr = ClusterizedEq(offsetIncr, ispace=ispace)
    itNodes.append(Increment(cOffsetIncr))
    
    itNodes.append(Call(name="lseek", arguments=[filesArray[tid], (-1)*offsetArray[tid], Macro("SEEK_END")]))
    itNodes.append(Call(name="read", arguments=[filesArray[tid], buffer, slices_size[tid, Slice]], retobj=ret))
    
    if1Nodes.append(Call(name="printf", arguments=[String("\"%zu\\n\""), offsetArray[tid]]))
    if1Nodes.append(Call(name="perror", arguments=[String("\"Cannot open output file\"")]))
    if1Nodes.append(Call(name="exit", arguments=1))
    itNodes.append(Conditional(CondNe(ret, slices_size[tid, Slice]), if1Nodes))
    
    if2Nodes.append(Call(name="printf", arguments=[String("\"decompression failed\\n\"")]))
    if2Nodes.append(Call(name="exit", arguments=1))
    zfpsize = Symbol(name="zfpsize", dtype=size_t)  # auxiliry
    itNodes.append(Call(name="zfp_decompress", arguments=[zfp, field], retobj=zfpsize))
    itNodes.append(Conditional(Not(zfpsize), if2Nodes))
    
    itNodes.append(Call(name="zfp_field_free", arguments=[field]))
    itNodes.append(Call(name="zfp_stream_close", arguments=[zfp]))
    itNodes.append(Call(name="stream_close", arguments=[stream]))
    itNodes.append(Call(name="free", arguments=[buffer]))
    
    newSptEq = IREq(sptArray[tid], (-1))
    cNewSptEq = ClusterizedEq(newSptEq, ispace=ispace)
    itNodes.append(Increment(cNewSptEq))
    
    decompressSection = Iteration(itNodes, uSizeDim, uVecSize1-1, direction=Backward, pragmas=[pragma])
    
    return Section("decompress", decompressSection)

def write_or_read_build(iet_body, is_forward, nthreads, files_dict, iSymbol, func_sizes_symb_dict, funcs_dict, t0, counters_dict, is_mpi):
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
    io_body=[]
    if is_forward:
        temp_name = "write_temp"
        name = "write"
        for func in funcs_dict:
            func_write = write_build(nthreads, files_dict[func], iSymbol, func_sizes_symb_dict[func],
                                      funcs_dict[func], t0, is_mpi)
            io_body.append(func_write)
        
    else: # gradient
        temp_name = "read_temp"
        name = "read"
        for func in funcs_dict:
            func_read = read_build(nthreads, files_dict[func], iSymbol, func_sizes_symb_dict[func],
                                    funcs_dict[func], t0, counters_dict[func])
            io_body.append(func_read)
          
    io_section = Section(name, io_body)
    update_iet(iet_body, temp_name, io_section)     


def write_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, is_mpi):
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
        Iteration: write loop
    """

    uVecSize1 = funcStencil.symbolic_shape[1]
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

    return Iteration(itNodes, uSizeDim, uVecSize1-1, pragmas=[pragma])

def read_build(nthreads, filesArray, iSymbol, func_size, funcStencil, t0, counters):
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
        Iteration: read loop
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
        
    return Iteration(itNodes, iDim, uVecSize1-1, direction=Backward, pragmas=[pragma])

def open_build(files_array_dict, counters_array_dict, metasArray, sptArray, offsetArray, nthreadsDim, nthreads, is_forward, iSymbol, ooc_compression, slices_size):
    """
    This method inteds to code open section for both Forward and Gradient operators.
    
    Args:
        filesArray (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        countersArray (counters): pointer of allocated memory of nthreads dimension. Each place has a size of int
        metasArray (Array): some array
        nthreadsDim (CustomDimension): dimension from 0 to nthreads 
        nthreads (NThreads): number of threads
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        ooc_compression (CompressionConfig): object representing compression settings

    Returns:
        Section: open section
    """
    
    # Build conditional
    # Regular Forward or Gradient
    arrays = [file_array for file_array in files_array_dict.values()] 
    if not ooc_compression and not is_forward:
        arrays.extend(counters_array for counters_array in counters_array_dict.values())
    # Compression Forward or Compression Gradient
    if ooc_compression:
        arrays.append(metasArray)
    if ooc_compression and not is_forward:
        arrays.extend([sptArray, offsetArray]) 

    arrays_cond = array_alloc_check(arrays) 
    
    #Call open_thread_files
    open_threads_calls = []
    for func_name in files_array_dict:
        funcArgs = [files_array_dict[func_name], nthreads, String('"{}"'.format(func_name))]
        if ooc_compression:
            funcArgs.append(metasArray)
        open_threads_calls.append(Call(name='open_thread_files_temp', arguments=funcArgs))

    # Open section body
    body = [arrays_cond, *open_threads_calls]
    
    # Additional initialization for Gradient operators
    if not is_forward and not ooc_compression:
        # Regular
        counters_init = []
        intervalGroup = IntervalGroup((Interval(nthreadsDim, 0, nthreads)))
        for counter in counters_array_dict.values():
            countersEq = ClusterizedEq(IREq(counter[iSymbol], 1), ispace=IterationSpace(intervalGroup))
            counters_init.append(Expression(countersEq, None, False))
        
        openIterationGrad = Iteration(counters_init, nthreadsDim, nthreads-1)
        body.append(openIterationGrad)
    
    elif not is_forward and ooc_compression:
        # Compression
        get_slices_size = Call(name="get_slices_size_temp", arguments=[metasArray, sptArray], retobj=slices_size)
        body.append(get_slices_size)

        intervalGroup = IntervalGroup((Interval(nthreadsDim, 0, nthreads)))
        c_offset_init_Eq = ClusterizedEq(IREq(offsetArray[iSymbol], 0), ispace=IterationSpace(intervalGroup))
        offset_init_eq = Expression(c_offset_init_Eq, None, False)
        closeCall = Call(name="close", arguments=[metasArray[iSymbol]])
        openIterationGrad = Iteration([offset_init_eq, closeCall], nthreadsDim, nthreads-1)
        body.append(openIterationGrad)
        
    return Section("open", body)

def close_build(nthreads, files_dict, iSymbol, nthreadsDim):
    """
    This method inteds to ls gradient.c close section.
    Obs: maybe the desciption of the variables should be better

    Args:
        nthreads (NThreads): symbol of number of threads
        files_dict (dict): dictionary with file pointers arrays
        iSymbol (Symbol): symbol of the iterator index i
        nthreadsDim (CustomDimension): dimension from 0 to nthreads

    Returns:
        section (Section): complete close section
    """

    # close(files[i]);
    itNodes=[]
    for func in files_dict:
        files = files_dict[func]
        itNodes.append(Call(name="close", arguments=[files[iSymbol]])) 
    
    # for(int i=0; i < nthreads; i++) --> for(int i=0; i <= nthreads-1; i+=1)
    closeIteration = Iteration(itNodes, nthreadsDim, nthreads-1)
    
    return Section("close", closeIteration)

def func_size_build(funcStencil, func_size, float_size):
    """
    Generates float_size init call and the init function size expression.

    Args:
        funcStencil (AbstractFunction): I/O function
        func_size (Symbol): Symbol representing the I/O function size
        float_size (Symbol): Symbol representing C "float" type size

    Returns:
        funcSizeExp: Expression initializing the function size
    """
    
    # TODO: Function name must come from user?
    sizes = funcStencil.symbolic_shape[2:]
    funcEq = IREq(func_size, (reduce(lambda x, y: x * y, sizes) * float_size))
    funcSizeExp = Expression(ClusterizedEq(funcEq, ispace=None), None, True)

    return funcSizeExp
