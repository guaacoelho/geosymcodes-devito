import numpy as np
import cgen
from sympy import Mod, Not
from pdb import set_trace
from ctypes import c_int32, POINTER, c_int, c_void_p
from functools import reduce 

from devito.tools import timed_pass
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, Macro, String, Null, Byref, cast_mapper, SizeOf)
from devito.symbolics.extended_sympy import Byref
from devito.types import (CustomDimension, Array, Symbol, Pointer, TimeDimension, PointerArray, ThreadID,
                          NThreads, off_t, zfp_type, size_t, zfp_field, bitstream, zfp_stream, Eq)
from devito.ir import (Expression, Increment, Iteration, List, Conditional, Call, Conditional, CallableBody, Callable,
                            Section, FindNodes, Transformer, Return, Definition, EntryFunction)
from devito.ir.iet.utils import ooc_array_alloc_check, ooc_update_iet, ooc_get_compress_mode_function
from devito.ir.equations import IREq, LoweredEq, ClusterizedEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward)



__all__ = ['ooc_build', 'ooc_efuncs']


def open_threads_build(nthreads, files_array, metas_array, i_symbol, nthreads_dim, name_array, is_write, is_mpi, is_compression, io_path):
    """
    This method generates the function open_thread_files according to the operator used.

    Args:
        nthreads (NThreads): number of threads
        files_array (Array): array of files
        metas_array (Array): some array
        i_symbol (Symbol): iterator symbol
        nthreads_dim (CustomDimension): dimension i from 0 to nthreads
        name_array (Array): Function name
        is_write (bool): True for the Forward operator; False for the Gradient operator
        is_mpi (bool): True for the use of MPI; False otherwise.
        is_compression (bool): True for the use of compression; False otherwise.

    Returns:
        Callable: the callable function open_thread_files
    """
    
    it_nodes=[]
    if_nodes=[]
    
    # TODO: initialize char name[100]
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32, ignoreDefinition=True)

    name_dim = [CustomDimension(name="name_dim", symbolic_size=100)]
    stencil_name_array = Array(name='stencil', dimensions=name_dim, dtype=np.byte)        
    
    if_nodes.append(Call(name="perror", arguments=String("\"Cannot open output file\\n\"")))
    if_nodes.append(Call(name="exit", arguments=1))
       
    if is_mpi:
        # TODO: initialize int myrank
        # TODO: initialize char error[140]
        myrank = Symbol(name="myrank", dtype=np.int32)
        mr_eq = IREq(myrank, 0)

        dps = Symbol(name="DPS", dtype=np.int32, ignoreDefinition=True)
        socket = Symbol(name="socket", dtype=np.int32)
        
        nvme_id_eq = IREq(nvme_id, Mod(i_symbol, ndisks)+socket)        
        socket_eq = IREq(socket, Mod(myrank, 2) * dps)
        c_socket_eq = ClusterizedEq(socket_eq, ispace=None)
        c_nvme_id_eq = ClusterizedEq(nvme_id_eq, ispace=None)                  
         
        it_nodes.append(Expression(ClusterizedEq(mr_eq), None, True))
        it_nodes.append(Call(name="MPI_Comm_rank", arguments=[Macro("MPI_COMM_WORLD"), Byref(myrank)]))
        it_nodes.append(Expression(c_socket_eq, None, True)) 
        it_nodes.append(Expression(c_nvme_id_eq, None, True)) 
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{io_path}/nvme%d/socket_%d_%s_vec_%d.bin\""), nvme_id, myrank, stencil_name_array, i_symbol]))
    else:
        nvme_id_eq = IREq(nvme_id, Mod(i_symbol, ndisks))
        c_nvme_id_eq = ClusterizedEq(nvme_id_eq, ispace=None)        
        it_nodes.append(Expression(c_nvme_id_eq, None, True))   
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{io_path}/nvme%d/%s_vec_%d.bin\""), nvme_id, stencil_name_array, i_symbol]))        
    
    op_flags = String("OPEN_FLAGS")
    o_flags_comp_write = String("O_WRONLY | O_CREAT | O_TRUNC")
    o_flags_comp_read = String("O_RDONLY")
    s_flags = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")        
    
    if is_write and is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_write, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{io_path}/nvme%d/%s_vec_%d.meta\""), nvme_id, stencil_name_array, i_symbol]))
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_write, s_flags], retobj=metas_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(metas_array[i_symbol], -1), if_nodes))
    elif is_write and not is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, op_flags, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
    elif not is_write and is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_read, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{io_path}/nvme%d/%s_vec_%d.meta\""), nvme_id, stencil_name_array, i_symbol]))
        it_nodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_read, s_flags], retobj=metas_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(metas_array[i_symbol], -1), if_nodes))
    elif not is_write and not is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, op_flags, s_flags], retobj=files_array[i_symbol]))   
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))    
    
    func_args = [files_array, nthreads, stencil_name_array]
    if is_compression:            
        func_args.append(metas_array)
    
    open_iteration = Iteration(it_nodes, nthreads_dim, nthreads)
    
    body = CallableBody(open_iteration)

    return Callable("open_thread_files", body, "void", func_args)

def get_slices_build(spt_array, nthreads, metas_array, nthreads_dim, i_symbol, slices_size):
    """_summary_

    Args:
        spt_array (Array): slices per thread array, for compression
        nthreads (NThreads): number of threads
        metas_array (Array): metas array for compression
        nthreads_dim (CustomDimension): dimension from 0 to nthreads
        i_symbol (Symbol): iterator symbol
        slices_size (PointerArray): 2d-array of slices, for compression

    Returns:
        _type_: _description_
    """
    
    it_nodes=[]
    if_nodes=[]
    func_body=[]
    
    malloc_call = Call(name="malloc", arguments=[nthreads*SizeOf(String(r"size_t *"))], 
                      retobj=Pointer(name='slices_size', dtype=POINTER(POINTER(size_t)), ignoreDefinition=True), cast=True)
    func_body.append(malloc_call)
    
    # Get size of the file
    f_size = Symbol(name='fsize', dtype=off_t)
    lseek_call = Call(name="lseek", arguments=[metas_array[i_symbol], cast_mapper[size_t](0), Macro("SEEK_END")], retobj=f_size)
    it_nodes.append(lseek_call)
    
    # Get number of slices per thread file
    spt_eq = IREq(spt_array[i_symbol], cast_mapper[int](f_size) / SizeOf(String(r"size_t")) -1)
    c_spt_eq = ClusterizedEq(spt_eq, ispace=None)
    it_nodes.append(Expression(c_spt_eq, None, False))
    
    # Allocate
    slices_size_tid_malloc_call = Call(name='(size_t *)malloc', arguments=[f_size], retobj=slices_size[i_symbol])
    it_nodes.append(slices_size_tid_malloc_call)
    
    if_nodes.append(Call(name="perror", arguments=String("\"Error to allocate slices\\n\"")))
    if_nodes.append(Call(name="exit", arguments=1))
    it_nodes.append(Conditional(CondEq(String(r"slices_size"), Macro("NULL")), if_nodes))
    
    # Return to begin of the file
    it_nodes.append(Call(name="lseek", arguments=[metas_array[i_symbol], 0, Macro("SEEK_SET")]))
    
    # Read to slices_size buffer
    it_nodes.append(Call(name="read", arguments=[metas_array[i_symbol], Byref(slices_size[i_symbol, 0]), f_size]))
    
    get_slices_iteration = Iteration(it_nodes, nthreads_dim, nthreads-1)
    func_body.append(get_slices_iteration)
    func_body.append(Return(String(r"slices_size")))
    
    return Callable("get_slices_size", CallableBody(func_body), "size_t**", [metas_array, spt_array, nthreads])    
    


def headers_build(is_write, is_compression, is_mpi, ooc_config):
    """
    Builds operator's headers

    Args:
        is_write (bool): True for the write mode; False for read mode
        is_compression (bool): True for the compression operator; False otherwise
        is_mpi (bool): True for MPI execution; False otherwise

    Returns:
        headers (List) : list with header defines
        includes (List): list with includes
    """
    ndisks = str(ooc_config.ndisks)
    dps = str(ooc_config.dps)
    _out_of_core_mpi_headers=[(("ifndef", "DPS"), ("DPS", dps))]
    _out_of_core_headers_write=[("_GNU_SOURCE", ""),
                                  (("ifndef", "NDISKS"), ("NDISKS", ndisks)), 
                                  (("ifdef", "CACHE"), ("OPEN_FLAGS", "O_WRONLY | O_CREAT"), ("else", ),
                                   ("OPEN_FLAGS", "O_DIRECT | O_WRONLY | O_CREAT"))]
    _out_of_core_headers_read=[("_GNU_SOURCE", ""),
                                   (("ifndef", "NDISKS"), ("NDISKS", ndisks)), 
                                   (("ifdef", "CACHE"), ("OPEN_FLAGS", "O_RDONLY"), ("else", ),
                                    ("OPEN_FLAGS", "O_DIRECT | O_RDONLY"))]
    _out_of_core_compression_headers=[(("ifndef", "NDISKS"), ("NDISKS", ndisks)),]
    _out_of_core_includes = ["fcntl.h", "stdio.h", "unistd.h"]
    _out_of_core_mpi_includes = ["mpi.h"]
    _out_of_core_compression_includes = ["zfp.h"]


    # Headers
    headers=[]
    if is_compression:
        headers.extend(_out_of_core_compression_headers)
    else: 
        if is_write:
            headers.extend(_out_of_core_headers_write)
        else:
            headers.extend(_out_of_core_headers_read)

        if is_mpi:
            headers.extend(_out_of_core_mpi_headers)

    # Includes
    includes=[]
    includes.extend(_out_of_core_includes)
    if is_mpi: includes.extend(_out_of_core_mpi_includes)
    if is_compression: includes.extend(_out_of_core_compression_includes)

    return headers, includes

@iet_pass
def ooc_efuncs(iet, **kwargs):
    """
    Orchestrates out of core efuncs build

    Args:
        iet (IterationTree): Iteration/Expression tree

    Returns:
        iet : transformed Iteration/Expression tree
        object: iet updated attributes 
    """
    # Out of core built only in the EntryFunction
    if not isinstance(iet, EntryFunction):
        return iet, {}
    
    ooc_config = kwargs['options']['out-of-core']
    is_write = ooc_config.mode == 'write'
    is_mpi = kwargs['options']['mpi']
    is_compression = ooc_config.compression
    efuncs = []
    mapper={}
    calls = FindNodes(Call).visit(iet)

    nthreads = NThreads(ignoreDefinition=True)
    name_dim = [CustomDimension(name="name_dim", symbolic_size=100)]
    name_array = Array(name='name', dimensions=name_dim, dtype=np.byte)
    i_symbol = Symbol(name="i", dtype=c_int32)

    nthreads_dim = CustomDimension(name="i", symbolic_size=nthreads) 
    files_array = Array(name='files', dimensions=[nthreads_dim], dtype=np.int32)
    metas_array = Array(name='metas', dimensions=[nthreads_dim], dtype=np.int32)

    if is_compression and not is_write:
        spt_array = Array(name='spt', dimensions=[nthreads_dim], dtype=np.int32)
        slices_size = PointerArray(name='slices_size', dimensions=[nthreads_dim], 
                                   array=Array(name='slices_size', dimensions=[nthreads_dim], dtype=size_t), ignoreDefinition=True)
        slices_size_callable = get_slices_build(spt_array, nthreads, metas_array, nthreads_dim, i_symbol, slices_size)
        efuncs.append(slices_size_callable)
        
        for call in calls:
            if call.name == 'get_slices_size_temp':
                new_get_slices_call = Call(name='get_slices_size', arguments=call.arguments,
                                           retobj=call.retobj)
                mapper[call] = new_get_slices_call

    open_threads_callable = open_threads_build(nthreads, files_array, metas_array, i_symbol,
                                             nthreads_dim, name_array, is_write, 
                                             is_mpi, is_compression, ooc_config.path)
    efuncs.append(open_threads_callable)   
    for call in calls:
        if call.name == 'open_thread_files_temp':
            new_open_thread_call = Call(name='open_thread_files', arguments=call.arguments)
            mapper[call] = new_open_thread_call
    
    iet = Transformer(mapper).visit(iet)   
    
    headers, includes = headers_build(is_write, is_compression, is_mpi, ooc_config)
    return iet, {'efuncs': efuncs, "headers": headers, "includes": includes}



@timed_pass(name='ooc_build')
def ooc_build(iet_body, ooc, nt, is_mpi, language, time_iterators):
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
    
    if is_mpi and len(funcs) > 1:
        raise ValueError("Multi Function currently does not support multi process")

    if is_mpi and ooc_compression:
        raise ValueError("Out of core currently does not support MPI and compression working togheter")
    
    funcs_dict = dict((func.name, func) for func in funcs)
    is_write = out_of_core == 'write'
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
    metas_dict= dict()
    spt_dict = dict()
    offset_dict = dict()
    slices_size_dict = dict()
    for func in funcs:
        metas_array = Array(name=func.name + '_metas', dimensions=[nthreads_dim], dtype=np.int32)
        spt_array = Array(name=func.name + '_spt', dimensions=[nthreads_dim], dtype=np.int32)
        offset_array = Array(name=func.name + '_offset', dimensions=[nthreads_dim], dtype=off_t)
        slices_size = PointerArray(name=func.name + '_slices_size', dimensions=(nthreads_dim, ), 
                               array=Array(name=func.name + '_slices_size', dimensions=[nthreads_dim], dtype=size_t, ignoreDefinition=True),
                               ignoreDefinition=True)
        
        metas_dict.update({func.name: metas_array})
        spt_dict.update({func.name: spt_array})
        offset_dict.update({func.name: offset_array})
        slices_size_dict.update({func.name: slices_size})


    ######## Build open section ########
    open_section = open_build(files_dict, counters_dict, metas_dict, spt_dict, offset_dict,
                             slices_size_dict, nthreads_dim, nthreads, is_write, i_symbol, ooc_compression)

    ######## Build func_size var ########
    float_size = Symbol(name="float_size", dtype=np.uint64)
    float_size_init = Call(name="sizeof", arguments=[String(r"float")], retobj=float_size)

    #TODO: create a DEVITO zfp_type_float to avoid using strings
    type_var = Symbol(name='type', dtype=zfp_type)
    
    func_sizes_dict = {}
    func_sizes_symb_dict={}
    for func in funcs:
        func_size = Symbol(name=func.name+"_size", dtype=np.uint64) 
        func_size_exp = func_size_build(func, func_size, float_size)
        func_sizes_dict.update({func.name: func_size_exp})
        func_sizes_symb_dict.update({func.name: func_size})

    if ooc_compression:                     
        ######## Build compress/decompress section ########
        compress_or_decompress_build(files_dict, metas_dict, iet_body, is_write, funcs_dict, nthreads,
                                     time_iterators, spt_dict, offset_dict, ooc_compression, slices_size_dict,type_var) 
    else:
        ######## Build write/read section ########    
        write_or_read_build(iet_body, is_write, nthreads, files_dict, func_sizes_symb_dict, funcs_dict,
                            time_iterator, counters_dict, is_mpi)
    
    
    ######## Build close section ########
    close_section = close_build(nthreads, files_dict, i_symbol, nthreads_dim)
    
    #TODO: Generate blank lines between sections
    for size_init in func_sizes_dict.values():
        iet_body.insert(0, size_init)
        
    if ooc_compression:
        type_eq = IREq(type_var, String(r"zfp_type_float"))
        c_type_eq = ClusterizedEq(type_eq, ispace=None)
        type_eq = Expression(c_type_eq, None, True)
        iet_body.insert(0, type_eq)
           
    iet_body.insert(0, float_size_init)
    iet_body.insert(0, open_section)
    iet_body.append(close_section)
    
    ######## Free slices memory ########
    if ooc_compression and not is_write:
        close_slices = close_slices_build(nthreads, i_symbol, slices_size_dict, nthreads_dim)
        iet_body.append(close_slices)
        
        for func in slices_size_dict:
            iet_body.append(Call(name="free", arguments=[(String(f"{func}_slices_size"))]))
        
    return iet_body


def open_build(files_array_dict, counters_array_dict, metas_dict, spt_dict, offset_dict, slices_size_dict, nthreads_dim, nthreads, is_write, i_symbol, ooc_compression):
    """
    This method builds open section for both Forward and Gradient operators.
    
    Args:
        files_array_dict (Dictionary): dict with files array of each Function
        counters_array_dict (Dictionary): dict with counters array of each Function
        metas_dict (Dictionary): dict with metas array of each Function, for compression
        spt_dict (Dictionary): dict with slices per thread array of each Function, for compression
        offset_dict (Dictionary): dict with offset array of each Function, for compression
        slices_size_dict (PointerArray): dict with 2d-array of slices of each Function, for compression
        nthreads_dim (CustomDimension): dimension from 0 to nthreads 
        nthreads (NThreads): number of threads
        is_write (bool): True for the Forward operator; False for the Gradient operator
        i_symbol (Symbol): iterator symbol
        ooc_compression (CompressionConfig): object representing compression settings

    Returns:
        Section: open section
    """
    
    # Build conditional
    # Regular Forward or Gradient
    arrays = [file_array for file_array in files_array_dict.values()] 
    if not ooc_compression and not is_write:
        arrays.extend(counters_array for counters_array in counters_array_dict.values())
    # Compression Forward or Compression Gradient
    if ooc_compression:
        arrays.extend(metas_array for metas_array in metas_dict.values())
    if ooc_compression and not is_write:
        arrays.extend(spt_array for spt_array in spt_dict.values())
        arrays.extend(offset_array for offset_array in offset_dict.values()) 

    arrays_cond = ooc_array_alloc_check(arrays) 
    
    #Call open_thread_files
    open_threads_calls = []
    for func_name in files_array_dict:
        func_args = [files_array_dict[func_name], nthreads, String('"{}"'.format(func_name))]
        if ooc_compression:
            func_args.append(metas_dict[func_name])
        open_threads_calls.append(Call(name='open_thread_files_temp', arguments=func_args))

    # Open section body
    body = [arrays_cond, *open_threads_calls]
    
    # Additional initialization for Gradient operators
    if not is_write and not ooc_compression:
        # Regular
        counters_init = []
        interval_group = IntervalGroup((Interval(nthreads_dim, 0, nthreads)))
        for counter in counters_array_dict.values():
            counters_eq = ClusterizedEq(IREq(counter[i_symbol], 1), ispace=IterationSpace(interval_group))
            counters_init.append(Expression(counters_eq, None, False))
        
        open_iteration_grad = Iteration(counters_init, nthreads_dim, nthreads-1)
        body.append(open_iteration_grad)
    
    elif not is_write and ooc_compression:
        # Compression
        get_slices_calls=[]
        offset_inits=[]
        close_metas=[]
        interval_group = IntervalGroup((Interval(nthreads_dim, 0, nthreads)))
        
        for func_name in files_array_dict:
            metas_str = String(f"{func_name}_metas_vec")
            spt_str = String(f"{func_name}_spt_vec")
            get_slices_size = Call(name='get_slices_size_temp', arguments=[metas_str, spt_str, nthreads], 
                        retobj=Pointer(name=f"{func_name}_slices_size", dtype=POINTER(POINTER(size_t)), ignoreDefinition=True))
            
            get_slices_calls.append(get_slices_size)
            
            offset_array = offset_dict[func_name]
            metas_array = metas_dict[func_name]
            c_offset_init_Eq = ClusterizedEq(IREq(offset_array[i_symbol], 0), ispace=IterationSpace(interval_group))
            offset_init_eq = Expression(c_offset_init_Eq, None, False)
            close_call = Call(name="close", arguments=[metas_array[i_symbol]])
            
            offset_inits.append(offset_init_eq)
            close_metas.append(close_call)
            
        body.extend(get_slices_calls)
        open_iteration_grad = Iteration(offset_inits + close_metas, nthreads_dim, nthreads-1)
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


def compress_or_decompress_build(files_dict, metas_dict, iet_body, is_write, funcs_dict, nthreads, time_iterators, spt_dict, offset_dict, ooc_compression, slices_dict, type_var):
    """
    This function decides if it is either a compression or a decompression

    Args:
        files_dict (Dictonary): dict with arrays of files
        metas_dict (Dictonary): dict with arrays of metadata
        iet_body (List): IET body nodes
        is_write (bool): if True, it is write. It is read otherwise
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        nthreads (NThreads): number of threads
        time_iterators (tuple): time iterator indexes
        spt_dict (Array): dict with arrays of slices per thread
        offset_dict (Array): dict with arrays of offset
        ooc_compression (CompressionConfig): object with compression settings
        slices_dict (PointerArray): dict with 2d-arrays of slices for compression mode
        type_var (Symbol): representation of zfp_type_float
    """
    
    sec_name = "compress" if is_write else "decompress"
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    iterations=[]
    eqs=[]
    for func in funcs_dict:
        func_size1 = funcs_dict[func].symbolic_shape[1]
        func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
        interval = Interval(func_size_dim, 0, func_size1)
        interval_group = IntervalGroup((interval))
        ispace = IterationSpace(interval_group)    
        
        tid = Symbol(name="tid", dtype=np.int32)
        i_symbol = Symbol(name="i", dtype=np.int32)
        tid_eq = IREq(tid, Mod(i_symbol, nthreads))
        c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
        
        if is_write:
            io_iteration = compress_build(files_dict[func], metas_dict[func], funcs_dict[func], i_symbol, pragma,
                                         func_size_dim, tid, c_tid_eq, ispace, time_iterators[0], ooc_compression, type_var)
        else:
            io_iteration = decompress_build(files_dict[func], funcs_dict[func], i_symbol, pragma, func_size_dim, tid, c_tid_eq,
                                ispace, time_iterators[-1], spt_dict[func], offset_dict[func], ooc_compression, slices_dict[func], type_var)
        
        iterations.append(io_iteration)
    
    io_section = Section(sec_name, iterations)

    ooc_update_iet(iet_body, sec_name + "_temp", io_section)      
    

def compress_build(files_array, metas_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq, ispace, t0, ooc_compression, type_var):
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
        type_var(Symbol): representation of zfp_type_float

    Returns:
        Section: compress section
    """
    
    func_size1 = func_stencil.symbolic_shape[1]    
    it_nodes=[]
    if_nodes=[]
    
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    field = Pointer(name="field", dtype=POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=size_t)
    buffer = Pointer(name="buffer", dtype=c_void_p)
    stream = Pointer(name="stream", dtype=POINTER(bitstream))
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
    
    io_iteration = Iteration(it_nodes, func_size_dim, func_size1-1, pragmas=[pragma])
    return io_iteration

def decompress_build(files_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq, ispace, t2, spt_array, offset_array, ooc_compression, slices_size, type_var):
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
        type_var(Symbol): representation of zfp_type_float

    Returns:
        Section: decompress section
    """
    
    func_size1 = func_stencil.symbolic_shape[1]    
    it_nodes=[]
    if1_nodes=[]
    if2_nodes=[]
    
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    field = Pointer(name="field", dtype=POINTER(zfp_field))
    zfp = Pointer(name="zfp", dtype=POINTER(zfp_stream))
    bufsize = Symbol(name="bufsize", dtype=off_t)
    buffer = Pointer(name="buffer", dtype=c_void_p)
    stream = Pointer(name="stream", dtype=POINTER(bitstream))
    slice_symbol = Symbol(name="slice", dtype=np.int32)
    ret = Symbol(name="ret", dtype=np.int32)
    
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
    
    io_iteration = Iteration(it_nodes, func_size_dim, func_size1-1, direction=Backward, pragmas=[pragma])
    
    return io_iteration

def write_or_read_build(iet_body, is_write, nthreads, files_dict, func_sizes_symb_dict, funcs_dict, t0, counters_dict, is_mpi):
    """
    Builds the read or write section of the operator, depending on the out_of_core mode.
    Replaces the temporary section at the end of the time iteration by the read or write section.   

    Args:
        iet_body (List): list of IET nodes 
        is_write (bool): True for the Forward operator; False for the Gradient operator
        nthreads (NThreads): symbol of number of threads
        files_dict (Dictonary): dict with arrays of files
        func_sizes_symb_dict (Dictonary): dict with Function sizes symbols
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        t0 (ModuloDimension): time t0
        counters_dict (Dictonary): dict with counter arrays for each Function
        is_mpi (bool): MPI execution flag

    """
    io_body=[]
    if is_write:
        temp_name = "write_temp"
        name = "write"
        for func in funcs_dict:
            func_write = write_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], t0, is_mpi)
            io_body.append(func_write)
        
    else: # read
        temp_name = "read_temp"
        name = "read"
        for func in funcs_dict:
            func_read = read_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], t0, counters_dict[func])
            io_body.append(func_read)
          
    io_section = Section(name, io_body)
    ooc_update_iet(iet_body, temp_name, io_section)     


def write_build(nthreads, files_array, func_size, func_stencil, t0, is_mpi):
    """
    This method inteds to code read.c write section.
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
    This method inteds to code read.c read section.
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
    This method inteds to ls read.c close section.
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


def close_slices_build(nthreads, i_symbol, slices_dict, nthreads_dim):
    """
    This method inteds to ls read.c free slices_size array memory.
    Obs: code creates variables that already exists on the previous code 

    Args:
        nthreads (NThreads): symbol of number of threads
        i_symbol (Symbol): iterator symbol
        slices_dict (Dictionary): dict with arrays of pointers to each compressed slice of each Function
        nthreads_dim (CustomDimension): dimension from 0 to nthreads
        
    Returns:
        close iteration (Iteration): close slices sizes iteration loop 
    """
    
    # free call for each Function;
    it_nodes=[]
    for slices_size in slices_dict.values():
        it_nodes.append(Call(name="free", arguments=[slices_size[i_symbol]]))    
    
    return Iteration(it_nodes, nthreads_dim, nthreads-1)