import numpy as np
import ctypes as ct
import cgen

from pdb import set_trace
from sympy import Mod, Not
from functools import reduce
from collections import OrderedDict

from devito.ir.iet.utils import ooc_array_alloc_check, ooc_update_iet, ooc_get_compress_mode_function
from devito.tools import timed_pass
from devito.symbolics import (CondNe, Macro, String, Null, Byref, SizeOf)
from devito.types.parallel import ThreadID
from devito.types import (CustomDimension, Array, Symbol, Pointer, TimeDimension, PointerArray,
                          NThreads, off_t, zfp_type, size_t, zfp_field, bitstream, zfp_stream, size_t, Eq)
from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Call, Conditional)
from devito.ir.equations import IREq, ClusterizedEq, LoweredEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward)

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

    if is_mpi and ooc_compression:
        raise ValueError("Out of core currently does not support MPI and compression working togheter")
    
    funcs_dict = dict((func.name, func) for func in funcs)
    is_forward = out_of_core == 'forward'
    time_iterator = time_iterators[0]

    ######## Dimension and symbol for iteration spaces ########
    nthreads = nt or NThreads()
    nthreads_dim = ThreadID(nthreads=nthreads)
    i_symbol = Symbol(name="tid", dtype=np.int32)
    

    ######## Build files and counters arrays ########
    files_dict = dict()
    counters_dict = dict()
    for func in funcs:
        files_array = Array(name=func.name + '_files', dimensions=[nthreads_dim], dtype=np.int32)
        counters_array = Array(name=func.name + '_counters', dimensions=[nthreads_dim], dtype=np.int32)
        files_dict.update({func.name: files_array})
        counters_dict.update({func.name: counters_array})

    # Compression arrays
    metas_array = Array(name='metas', dimensions=[nthreads_dim], dtype=np.int32)
    spt_array = Array(name='spt', dimensions=[nthreads_dim], dtype=np.int32)
    offset_array = Array(name='offset', dimensions=[nthreads_dim], dtype=off_t)
    slices_size = PointerArray(name='slices_size', dimensions=(nthreads_dim, ), 
                               array=Array(name='slices_size', dimensions=[nthreads_dim], dtype=size_t, ignoreDefinition=True),
                               ignoreDefinition=True)

    ######## Build open section ########
    open_section = open_build(files_dict, counters_dict, metas_array, spt_array, offset_array,
                             nthreads_dim, nthreads, is_forward, i_symbol, ooc_compression, slices_size)

    ######## Build func_size var ########
    float_size = Symbol(name="float_size", dtype=np.uint64)
    float_size_init = Call(name="sizeof", arguments=[String(r"float")], retobj=float_size)

    func_sizes_dict = {}
    func_sizes_symb_dict={}
    for func in funcs:
        func_size = Symbol(name=func.name+"_size", dtype=np.uint64) 
        func_size_exp = func_size_build(func, func_size, float_size)
        func_sizes_dict.update({func.name: func_size_exp})
        func_sizes_symb_dict.update({func.name: func_size})

    if ooc_compression:                     
        ######## Build compress/decompress section ########
        compress_or_decompress_build(files_dict, metas_array, iet_body, is_forward, funcs_dict, nthreads,
                                     time_iterators, spt_array, offset_array, ooc_compression, slices_size) 
    else:
        ######## Build write/read section ########    
        write_or_read_build(iet_body, is_forward, nthreads, files_dict, func_sizes_symb_dict, funcs_dict,
                            time_iterator, counters_dict, is_mpi)
    
    
    ######## Build close section ########
    close_section = close_build(nthreads, files_dict, i_symbol, nthreads_dim)
    
    #TODO: Generate blank lines between sections
    for size_init in func_sizes_dict.values():
        iet_body.insert(0, size_init)
    iet_body.insert(0, float_size_init)
    iet_body.insert(0, open_section)
    iet_body.append(close_section)
    
    ######## Free slices memory ########
    if ooc_compression and not is_forward:
        close_slices = close_slices_build(nthreads, i_symbol, slices_size, nthreads_dim)
        iet_body.append(close_slices)
        iet_body.append(Call(name="free", arguments=[(String(r"slices_size"))]))
        
    return iet_body


def open_build(files_array_dict, counters_array_dict, metas_array, spt_array, offset_array, nthreads_dim, nthreads, is_forward, i_symbol, ooc_compression, slices_size):
    """
    This method builds open section for both Forward and Gradient operators.
    
    Args:
        files_array_dict (Dictionary): dict with files array of each Function
        counters_array_dict (Dictionary): dict with counters array of each Function
        metas_array (Array): metas array for compression
        spt_array (Array): slices per thread array, for compression
        offset_array (Array): offset array, for compression
        nthreads_dim (CustomDimension): dimension from 0 to nthreads 
        nthreads (NThreads): number of threads
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        i_symbol (Symbol): iterator symbol
        ooc_compression (CompressionConfig): object representing compression settings
        slices_size (PointerArray): 2d-array of slices

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
        arrays.append(metas_array)
    if ooc_compression and not is_forward:
        arrays.extend([spt_array, offset_array]) 

    arrays_cond = ooc_array_alloc_check(arrays) 
    
    #Call open_thread_files
    open_threads_calls = []
    for func_name in files_array_dict:
        func_args = [files_array_dict[func_name], nthreads, String('"{}"'.format(func_name))]
        if ooc_compression:
            func_args.append(metas_array)
        open_threads_calls.append(Call(name='open_thread_files_temp', arguments=func_args))

    # Open section body
    body = [arrays_cond, *open_threads_calls]
    
    # Additional initialization for Gradient operators
    if not is_forward and not ooc_compression:
        # Regular
        counters_init = []
        interval_group = IntervalGroup((Interval(nthreads_dim, 0, nthreads)))
        for counter in counters_array_dict.values():
            counters_eq = ClusterizedEq(IREq(counter[i_symbol], 1), ispace=IterationSpace(interval_group))
            counters_init.append(Expression(counters_eq, None, False))
        
        open_iteration_grad = Iteration(counters_init, nthreads_dim, nthreads-1)
        body.append(open_iteration_grad)
    
    elif not is_forward and ooc_compression:
        # Compression
        get_slices_size = Call(name='get_slices_size_temp', arguments=[String(r"metas_vec"), String(r"spt_vec"), nthreads], 
                               retobj=Pointer(name='slices_size', dtype=ct.POINTER(ct.POINTER(size_t)), ignoreDefinition=True))
        body.append(get_slices_size)

        interval_group = IntervalGroup((Interval(nthreads_dim, 0, nthreads)))
        c_offset_init_Eq = ClusterizedEq(IREq(offset_array[i_symbol], 0), ispace=IterationSpace(interval_group))
        offset_init_eq = Expression(c_offset_init_Eq, None, False)
        close_call = Call(name="close", arguments=[metas_array[i_symbol]])
        open_iteration_grad = Iteration([offset_init_eq, close_call], nthreads_dim, nthreads-1)
        body.append(open_iteration_grad)
        
    return Section("open", body)


def func_size_build(func_stencil, func_size, float_size):
    """
    Generates float_size init call and the init function size expression.

    Args:
        func_stencil (AbstractFunction): I/O function
        func_size (Symbol): Symbol representing the I/O function size
        float_size (Symbol): Symbol representing C "float" type size

    Returns:
        func_size_exp: Expression initializing the function size
    """
    
    sizes = func_stencil.symbolic_shape[2:]
    func_eq = IREq(func_size, (reduce(lambda x, y: x * y, sizes) * float_size))
    func_size_exp = Expression(ClusterizedEq(func_eq, ispace=None), None, True)

    return func_size_exp


def compress_or_decompress_build(files_dict, metas_array, iet_body, is_forward, funcs_dict, nthreads, time_iterators, spt_array, offset_array, ooc_compression, slices_size):
    """
    This function decides if it is either a compression or a decompression

    Args:
        files_dict (Dictonary): dict with arrays of files
        metas_dict (Dictonary): dict with arrays of metadata
        iet_body (List): IET body nodes
        is_forward (bool): if True, it is forward. It is gradient otherwise
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        nthreads (NThreads): number of threads
        time_iterators (tuple): time iterator indexes
        spt_array (Array): array of slices per thread
        offset_array (Array): array of offset
        ooc_compression (CompressionConfig): object with compression settings
        slices_size (PointerArray): 2d-array of slices for compression mode
    """
    
    # TODO: Temporary workaround, while compression mode supports only one Function
    func_stencil = next(iter(funcs_dict.values()))
    files_array = next(iter(files_dict.values()))

    func_size1 = func_stencil.symbolic_shape[1]
    func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
    interval = Interval(func_size_dim, 0, func_size1)
    interval_group = IntervalGroup((interval))
    ispace = IterationSpace(interval_group)    
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    
    tid = Symbol(name="tid", dtype=np.int32)
    i_symbol = Symbol(name="i", dtype=np.int32)
    tid_eq = IREq(tid, Mod(i_symbol, nthreads))
    c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
    
    if is_forward:
        ooc_section = compress_build(files_array, metas_array, func_stencil, i_symbol, pragma,
                                     func_size_dim, tid, c_tid_eq, ispace, time_iterators[0], ooc_compression)
        temp_name = "compress_temp"
    else:
        ooc_section = decompress_build(files_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq,
                                       ispace, time_iterators[-1], spt_array, offset_array, ooc_compression, slices_size)
        temp_name = "decompress_temp"

    ooc_update_iet(iet_body, temp_name, ooc_section)      
    

def compress_build(files_array, metas_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq, ispace, t0, ooc_compression):
    """
    This function generates compress section.

    Args:
        files_array (Array): array of files
        metas_array (Array): array of metadata
        func_stencil (Function): Function defined by user for Operator
        i_symbol (Symbol): iterator symbol
        pragma (Pragma): omp pragma directives
        func_size_dim (CustomDimension): symbolic dimension of loop
        tid (Symbol): iterator index symbol
        c_tid_eq (ClusterizedEq): expression that defines tid --> int tid = i%nthreads
        ispace (IterationSpace): space of iteration
        t0 (ModuloDimension): time iterator index for compression
        ooc_compression (CompressionConfig): object with compression settings

    Returns:
        Section: compress section
    """
    
    func_size1 = func_stencil.symbolic_shape[1]    
    it_nodes=[]
    if_nodes=[]
    
    type_var = Symbol(name='type', dtype=zfp_type)
    type_eq = IREq(type_var, String(r"zfp_type_float"))
    c_type_eq = ClusterizedEq(type_eq, ispace=ispace)
    
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    field = Pointer(name="field", dtype=ct.POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=ct.POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=size_t)
    buffer = Pointer(name="buffer", dtype=ct.c_void_p)
    stream = Pointer(name="stream", dtype=ct.POINTER(bitstream))
    zfpsize = Symbol(name="zfpsize", dtype=size_t)
    
    it_nodes.append(Call(name="zfp_field_2d", arguments=[func_stencil[t0,i_symbol], type_var, func_stencil.symbolic_shape[2], func_stencil.symbolic_shape[3]], retobj=field))
    it_nodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    it_nodes.append(ooc_get_compress_mode_function(ooc_compression, zfp, field, type_var))
    it_nodes.append(Call(name="zfp_stream_maximum_size", arguments=[zfp, field], retobj=bufsize))
    it_nodes.append(Call(name="malloc", arguments=[bufsize], retobj=buffer))
    it_nodes.append(Call(name="stream_open", arguments=[buffer, bufsize], retobj=stream))
    it_nodes.append(Call(name="zfp_stream_set_bit_stream", arguments=[zfp, stream]))
    it_nodes.append(Call(name="zfp_stream_rewind", arguments=[zfp]))
    it_nodes.append(Call(name="zfp_compress", arguments=[zfp, field], retobj=zfpsize))
    if_nodes.append(Call(name="fprintf", arguments=[String(r"stderr"), String("\"compression failed\\n\"")]))
    if_nodes.append(Call(name="exit", arguments=1))
    it_nodes.append(Conditional(Not(zfpsize), if_nodes))
    
    it_nodes.append(Call(name="write", arguments=[files_array[tid], buffer, zfpsize]))
    it_nodes.append(Call(name="write", arguments=[metas_array[tid], Byref(zfpsize), SizeOf(String(r"size_t"))]))
    it_nodes.append(Call(name="zfp_field_free", arguments=[field]))
    it_nodes.append(Call(name="zfp_stream_close", arguments=[zfp]))
    it_nodes.append(Call(name="stream_close", arguments=[stream]))
    it_nodes.append(Call(name="free", arguments=[buffer]))
    
    compress_section = [Expression(c_type_eq, None, True), Iteration(it_nodes, func_size_dim, func_size1-1, pragmas=[pragma])]
    return Section("compress", compress_section)

def decompress_build(files_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq, ispace, t2, spt_array, offset_array, ooc_compression, slices_size):
    """
    This function generates decompress section.

    Args:
        files_array (Array): array of files
        func_stencil (Function): Function defined by user for Operator
        i_symbol (Symbol): iterator symbol
        pragma (Pragma): omp pragma directives
        func_size_dim (CustomDimension): symbolic dimension of loop
        tid (Symbol): iterator index symbol
        c_tid_eq (ClusterizedEq): expression that defines tid --> int tid = i%nthreads
        ispace (IterationSpace): space of iteration
        t2 (ModuloDimension): time iterator index for compression
        spt_array (Array): array of slices per thread
        offset_array (Array): array of offset
        ooc_compression (CompressionConfig): object with compression settings
        slices_size (PointerArray): 2d-array of slices

    Returns:
        Section: decompress section
    """
    
    func_size1 = func_stencil.symbolic_shape[1]    
    it_nodes=[]
    if1_nodes=[]
    if2_nodes=[]
    
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    type_var = Symbol(name='type', dtype=zfp_type)
    field = Pointer(name="field", dtype=ct.POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=ct.POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=off_t)
    buffer = Pointer(name="buffer", dtype=ct.c_void_p)
    stream = Pointer(name="stream", dtype=ct.POINTER(bitstream))
    slice_symbol = Symbol(name="slice", dtype=np.int32)
    ret = Symbol(name="ret", dtype=np.int32)
    
    # type_eq = IREq(type_var, Symbol(name="zfp_type_float", dtype=ct.c_int))
    type_eq = IREq(type_var, String(r"zfp_type_float"))
    c_type_eq = ClusterizedEq(type_eq, ispace=ispace)
    it_nodes.append(Expression(c_type_eq, None, True))
    
    it_nodes.append(Call(name="zfp_field_2d", arguments=[func_stencil[t2,i_symbol], type_var, func_stencil.symbolic_shape[2], func_stencil.symbolic_shape[3]], retobj=field))
    it_nodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    it_nodes.append(ooc_get_compress_mode_function(ooc_compression, zfp, field, type_var))
    it_nodes.append(Call(name="zfp_stream_maximum_size", arguments=[zfp, field], retobj=bufsize))
    it_nodes.append(Call(name="malloc", arguments=[bufsize], retobj=buffer))
    it_nodes.append(Call(name="stream_open", arguments=[buffer, bufsize], retobj=stream))
    it_nodes.append(Call(name="zfp_stream_set_bit_stream", arguments=[zfp, stream]))
    it_nodes.append(Call(name="zfp_stream_rewind", arguments=[zfp]))
    
    slice_eq = IREq(slice_symbol, spt_array[tid])
    c_slice_eq = ClusterizedEq(slice_eq, ispace=ispace)
    it_nodes.append(Expression(c_slice_eq, None, True))
    
    offset_incr = IREq(offset_array[tid], slices_size[tid, slice_symbol])
    c_offset_incr = ClusterizedEq(offset_incr, ispace=ispace)
    it_nodes.append(Increment(c_offset_incr))
    
    it_nodes.append(Call(name="lseek", arguments=[files_array[tid], (-1)*offset_array[tid], Macro("SEEK_END")]))
    it_nodes.append(Call(name="read", arguments=[files_array[tid], buffer, slices_size[tid, slice_symbol]], retobj=ret))
    
    if1_nodes.append(Call(name="printf", arguments=[String("\"%zu\\n\""), offset_array[tid]]))
    if1_nodes.append(Call(name="perror", arguments=[String("\"Cannot open output file\"")]))
    if1_nodes.append(Call(name="exit", arguments=1))
    it_nodes.append(Conditional(CondNe(ret, slices_size[tid, slice_symbol]), if1_nodes))
    
    if2_nodes.append(Call(name="printf", arguments=[String("\"decompression failed\\n\"")]))
    if2_nodes.append(Call(name="exit", arguments=1))
    zfpsize = Symbol(name="zfpsize", dtype=size_t)  # auxiliry
    it_nodes.append(Call(name="zfp_decompress", arguments=[zfp, field], retobj=zfpsize))
    it_nodes.append(Conditional(Not(zfpsize), if2_nodes))
    
    it_nodes.append(Call(name="zfp_field_free", arguments=[field]))
    it_nodes.append(Call(name="zfp_stream_close", arguments=[zfp]))
    it_nodes.append(Call(name="stream_close", arguments=[stream]))
    it_nodes.append(Call(name="free", arguments=[buffer]))
    
    new_spt_eq = IREq(spt_array[tid], (-1))
    c_new_spt_eq = ClusterizedEq(new_spt_eq, ispace=ispace)
    it_nodes.append(Increment(c_new_spt_eq))
    
    decompress_section = Iteration(it_nodes, func_size_dim, func_size1-1, direction=Backward, pragmas=[pragma])
    
    return Section("decompress", decompress_section)

def write_or_read_build(iet_body, is_forward, nthreads, files_dict, func_sizes_symb_dict, funcs_dict, t0, counters_dict, is_mpi):
    """
    Builds the read or write section of the operator, depending on the out_of_core mode.
    Replaces the temporary section at the end of the time iteration by the read or write section.   

    Args:
        iet_body (List): list of IET nodes 
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        nthreads (NThreads): symbol of number of threads
        files_dict (Dictonary): dict with arrays of files
        func_sizes_symb_dict (Dictonary): dict with Function sizes symbols
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        t0 (ModuloDimension): time t0
        counters_dict (Dictonary): dict with counter arrays for each Function
        is_mpi (bool): MPI execution flag

    """
    io_body=[]
    if is_forward:
        temp_name = "write_temp"
        name = "write"
        for func in funcs_dict:
            func_write = write_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], t0, is_mpi)
            io_body.append(func_write)
        
    else: # gradient
        temp_name = "read_temp"
        name = "read"
        for func in funcs_dict:
            func_read = read_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], t0, counters_dict[func])
            io_body.append(func_read)
          
    io_section = Section(name, io_body)
    ooc_update_iet(iet_body, temp_name, io_section)     


def write_build(nthreads, files_array, func_size, func_stencil, t0, is_mpi):
    """
    This method inteds to code gradient.c write section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        files_array (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        func_size (Symbol): the func_stencil size
        func_stencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        is_mpi (bool): MPI execution flag

    Returns:
        Iteration: write loop
    """

    func_size1 = func_stencil.symbolic_shape[1]
    func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
    interval = Interval(func_size_dim, 0, func_size1)
    interval_group = IntervalGroup((interval))
    ispace = IterationSpace(interval_group)
    it_nodes = []

    tid = Symbol(name="tid", dtype=np.int32)
    tid_eq = IREq(tid, Mod(func_size_dim, nthreads))
    c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    ret = Symbol(name="ret", dtype=np.int32)
    write_call = Call(name="write", arguments=[files_array[tid], func_stencil[t0, func_size_dim], func_size], retobj=ret)
    it_nodes.append(write_call)
    
    pstring = String("\"Write size mismatch with function slice size\"")

    cond_nodes = [Call(name="perror", arguments=pstring)]
    cond_nodes.append(Call(name="exit", arguments=1))
    cond = Conditional(CondNe(ret, func_size), cond_nodes)
    it_nodes.append(cond)

    # TODO: Pragmas should depend on the user's selected optimization options and be generated by the compiler
    if is_mpi:
        pragma = cgen.Pragma("omp parallel for schedule(static,1)")
    else:
        pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")

    return Iteration(it_nodes, func_size_dim, func_size1-1, pragmas=[pragma])

def read_build(nthreads, files_array, func_size, func_stencil, t0, counters):
    """
    This method inteds to code gradient.c read section.
    Obs: maybe the desciption of the variables should be better    

    Args:
        nthreads (NThreads): symbol of number of threads
        files_array (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        func_size (Symbol): the func_stencil size
        func_stencil (u): a stencil we call u
        t0 (ModuloDimension): time t0
        counters (array): pointer of allocated memory of nthreads dimension. Each place has a size of int

    Returns:
        Iteration: read loop
    """
    
    func_size1 = func_stencil.symbolic_shape[1]
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    i_dim = CustomDimension(name="i", symbolic_size=func_size1)
    interval = Interval(i_dim, 0, func_size1)
    interval_group = IntervalGroup((interval))
    ispace = IterationSpace(interval_group)
    it_nodes = []

    # Initialize tid
    tid = Symbol(name="tid", dtype=np.int32)
    tid_eq = IREq(tid, Mod(i_dim, nthreads))
    c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    # Build offset and call to lseek
    offset = Symbol(name="offset", dtype=off_t)
    offset_eq = IREq(offset, (-1)*counters[tid]*func_size)
    c_offset_eq = ClusterizedEq(offset_eq, ispace=ispace)
    it_nodes.append(Expression(c_offset_eq, None, True))    
    it_nodes.append(Call(name="lseek", arguments=[files_array[tid], offset, Macro("SEEK_END")]))

    # Initialize ret
    ret = Symbol(name="ret", dtype=np.int32)
    read_call = Call(name="read", arguments=[files_array[tid], func_stencil[t0, i_dim], func_size], retobj=ret)
    it_nodes.append(read_call)

    # Error conditional print
    pstring = String("\"Cannot open output file\"")
    cond_nodes = [
        Call(name="printf", arguments=[String("\"%d\""), ret]),
        Call(name="perror", arguments=pstring), 
        Call(name="exit", arguments=1)
    ]
    cond = Conditional(CondNe(ret, func_size), cond_nodes) # if (ret != func_size)
    it_nodes.append(cond)
    
    # Counters increment
    newCountersEq = IREq(counters[tid], 1)
    c_new_counters_eq = ClusterizedEq(newCountersEq, ispace=ispace)
    it_nodes.append(Increment(c_new_counters_eq))
        
    return Iteration(it_nodes, i_dim, func_size1-1, direction=Backward, pragmas=[pragma])


def close_build(nthreads, files_dict, i_symbol, nthreads_dim):
    """
    This method inteds to ls gradient.c close section.
    Obs: maybe the desciption of the variables should be better

    Args:
        nthreads (NThreads): symbol of number of threads
        files_dict (dict): dictionary with file pointers arrays
        i_symbol (Symbol): symbol of the iterator index i
        nthreads_dim (CustomDimension): dimension from 0 to nthreads

    Returns:
        Section: complete close section
    """

    it_nodes=[]
    
    for func in files_dict:
        files = files_dict[func]
        it_nodes.append(Call(name="close", arguments=[files[i_symbol]])) 
    
    close_iteration = Iteration(it_nodes, nthreads_dim, nthreads-1)
    
    return Section("close", close_iteration)


def close_slices_build(nthreads, i_symbol, slices_size, nthreads_dim):
    """
    This method inteds to ls gradient.c free slices_size array memory.
    Obs: code creates variables that already exists on the previous code 

    Args:
        nthreads (NThreads): symbol of number of threads
        i_symbol (Symbol): iterator symbol
        slices_size (PointerArray): array of pointers to each compressed slice
        nthreads_dim (CustomDimension): dimension from 0 to nthreads
        
    Returns:
        close iteration (Iteration): close slices sizes iteration loop 
    """
    
    # free(slices_size[i]);
    it_node = Call(name="free", arguments=[slices_size[i_symbol]])    
    
    # for(int tid=0; tid < nthreads; i++) --> for(int tid=0; tid <= nthreads-1; tid+=1)
    return Iteration(it_node, nthreads_dim, nthreads-1)
