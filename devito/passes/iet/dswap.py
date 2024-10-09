import os
import pwd
import numpy as np
import cgen

from socket import gethostname
from sympy import Mod, Not, sympify
from pdb import set_trace
from ctypes import c_int32, POINTER, c_int, c_void_p
from functools import reduce 
from mpi4py import MPI

from devito.tools import timed_pass
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, Macro, String, Null, Byref, cast_mapper, SizeOf)
from devito.symbolics.extended_sympy import Byref
from devito.types import (CustomDimension, Array, Symbol, Pointer, TimeDimension, PointerArray, ThreadID,
                          Indexed, NThreads, off_t, zfp_type, size_t, zfp_field, bitstream, zfp_stream, Eq)
from devito.ir import (Expression, Increment, Iteration, List, Conditional, Call, Conditional, CallableBody, Callable,
                            Section, FindNodes, FindSymbols, Transformer, Return, Definition, EntryFunction, Pragma)
from devito.ir.iet.utils import (dswap_array_alloc_check, dswap_update_iet, dswap_get_compress_mode_function,
                                 dswap_get_read_time_iterator)
from devito.ir.equations import IREq, LoweredEq, ClusterizedEq
from devito.ir.support import (Interval, IntervalGroup, IterationSpace, Backward)



__all__ = ['disk_swap_build', 'disk_swap_efuncs']


def open_threads_build(nthreads, files_array, metas_array, i_symbol, nthreads_dim, name_array, is_write, is_mpi, is_compression, io_path, io_folder):
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
    
    if io_folder:
        exec_id_path = f"{io_path}/{io_folder}"
    else:
        # Get pid
        try:
            pid = os.getppid() if is_mpi else os.getpid()
        except:
            raise RuntimeError(
                "Unable to retrieve pid (os.getpid()) during folder creation. Please, provide a folder name for the output files.")
        
        # Get host name
        try:
            host_name = gethostname()
        except:
            raise RuntimeError(
                "Unable to retrieve host name (socket.gethostname) during folder creation. Please, provide a folder name for the output files.")
        
        # Get username
        try:
           user_name = os.getlogin()
        except:
            try:
               user_name = pwd.getpwuid(os.getuid())[0]
            except:
                raise RuntimeError(
                    "Unable to retrieve username (os.getlogin() or os.getpwuid(os.getuid())[0]) during folder creation. Please, provide a folder name for the output files.")

        exec_id = "_".join([host_name, user_name, str(pid)])
        exec_id_path = f"{io_path}/{exec_id}"
    
    rank = MPI.COMM_WORLD.Get_rank() if is_mpi else 0
    
    if rank == 0 and not os.path.exists(exec_id_path):
        os.makedirs(exec_id_path)

    name_dim = [CustomDimension(name="name_dim", symbolic_size=100)]
    stencil_name_array = Array(name='stencil', dimensions=name_dim, dtype=np.byte)        
    
    if_nodes.append(Call(name="perror", arguments=String("\"Cannot open output file\\n\"")))
    if_nodes.append(Call(name="exit", arguments=1))
       
    if is_mpi:
        # TODO: initialize int myrank
        # TODO: initialize char error[140]
        myrank = Symbol(name="myrank", dtype=np.int32)
        mr_eq = IREq(myrank, 0)              
         
        it_nodes.append(Expression(ClusterizedEq(mr_eq), None, True))
        it_nodes.append(Call(name="MPI_Comm_rank", arguments=[Macro("MPI_COMM_WORLD"), Byref(myrank)]))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{exec_id_path}/socket_%d_%s_vec_%d.bin\""), myrank, stencil_name_array, i_symbol]))
    else:   
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{exec_id_path}/%s_vec_%d.bin\""), stencil_name_array, i_symbol]))        
    
    op_flags = String("OPEN_FLAGS")
    o_flags_comp_write = String("O_WRONLY | O_CREAT | O_TRUNC")
    o_flags_comp_read = String("O_RDONLY")
    s_flags = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")        
    
    if is_write and is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_write, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{exec_id_path}/%s_vec_%d.meta\""), stencil_name_array, i_symbol]))
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
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String(f"\"{exec_id_path}/%s_vec_%d.meta\""), stencil_name_array, i_symbol]))
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
    
    open_iteration = Iteration(it_nodes, nthreads_dim, nthreads-1)
    
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
    lseek_call = Call(name="lseek", arguments=[metas_array[i_symbol], 0, Macro("SEEK_END")], retobj=f_size)
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
    


def headers_build(is_write, is_compression, is_mpi, dswap_config):
    """
    Builds operator's headers

    Args:
        is_write (bool): True for the write mode; False for read mode
        is_compression (bool): True for the compression operator; False otherwise
        is_mpi (bool): True for MPI execution; False otherwise
        dswap_config (DiskSwapConfig): dswap configuration

    Returns:
        headers (List) : list with header defines
        includes (List): list with includes
    """
    odirect = "O_DIRECT | " if dswap_config.odirect else str()
    
    open_flags = "O_WRONLY | O_CREAT" if is_write else "O_RDONLY"
    
    _disk_swap_headers=[("_GNU_SOURCE", ""),
                          ("OPEN_FLAGS", odirect + open_flags)]
    _disk_swap_includes = ["fcntl.h", "stdio.h", "unistd.h"]
    _disk_swap_mpi_includes = ["mpi.h"]
    _disk_swap_compression_includes = ["zfp.h"]


    # Headers
    headers=[]
    if not is_compression: 
        headers.extend(_disk_swap_headers)

    # Includes
    includes=[]
    includes.extend(_disk_swap_includes)
    if is_mpi: includes.extend(_disk_swap_mpi_includes)
    if is_compression: includes.extend(_disk_swap_compression_includes)

    return headers, includes

@iet_pass
def disk_swap_efuncs(iet, **kwargs):
    """
    Orchestrates disk swap efuncs build

    Args:
        iet (IterationTree): Iteration/Expression tree

    Returns:
        iet : transformed Iteration/Expression tree
        object: iet updated attributes 
    """
    # Disk swap built only in the EntryFunction
    if not isinstance(iet, EntryFunction):
        return iet, {}
    
    dswap_config = kwargs['options']['disk-swap']
    is_write = dswap_config.mode == 'write'
    is_mpi = kwargs['options']['mpi']
    is_compression = dswap_config.compression
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
                                             is_mpi, is_compression, dswap_config.path, dswap_config.folder)
    efuncs.append(open_threads_callable)   
    for call in calls:
        if call.name == 'open_thread_files_temp':
            new_open_thread_call = Call(name='open_thread_files', arguments=call.arguments)
            mapper[call] = new_open_thread_call
    
    iet = Transformer(mapper).visit(iet)   
    
    headers, includes = headers_build(is_write, is_compression, is_mpi, dswap_config)
    return iet, {'efuncs': efuncs, "headers": headers, "includes": includes}



@timed_pass(name='disk_swap_build')
def disk_swap_build(iet_body, dswap, nt, is_mpi, language, time_iterators):
    """
    This private method builds a iet_body (list) with disk-swap nodes.

    Args:
        iet_body (List): a list of nodes
        dswap (Object): disk swap parameters
        nt (NThreads): symbol representing nthreads parameter of OpenMP
        is_mpi (bool): MPI execution flag
        language (str): language set for the operator (C, openmp or openacc)
        time_iterators(Dimension): iterator used as index in each timestep

    Returns:
        List : iet_body is a list of nodes
    """
    
    funcs = dswap.functions
    disk_swap = dswap.mode
    dswap_compression = dswap.compression    

    if language != 'openmp':
       raise ValueError("Disk swap requires OpenMP. Language parameter must be openmp, got %s" % language)
    
    for func in funcs:
        if func.save:
            raise ValueError("Disk swap incompatible with TimeFunction save functionality on %s" % func.name)

    if is_mpi and dswap_compression:
        raise NotImplementedError("Disk swap currently does not support MPI and compression working togheter")
    
    funcs_dict = dict((func.name, func) for func in funcs)
    is_write = disk_swap == 'write'

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
                             slices_size_dict, nthreads_dim, nthreads, is_write, i_symbol, dswap_compression)

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

    ######## Build initial offload or initial offset section ########
    if is_write:
        pre_time_io_section = init_offload_build(funcs_dict, nthreads, files_dict, func_sizes_symb_dict, is_mpi,
                                                 metas_dict, dswap_compression, type_var)

    if not is_write:
        pre_time_io_section = init_offset_build(funcs_dict, nthreads, counters_dict, dswap_compression,
                                                spt_dict, offset_dict, slices_size_dict)
    
    write_iterator = time_iterators[-1]
    if dswap_compression:                     
        ######## Build compress/decompress section ########
        compress_or_decompress_build(files_dict, metas_dict, iet_body, is_write, funcs_dict, nthreads,
                                     write_iterator, spt_dict, offset_dict, dswap_compression, slices_size_dict,type_var) 
    else:
        ######## Build write/read section ########    
        write_or_read_build(iet_body, is_write, nthreads, files_dict, func_sizes_symb_dict, funcs_dict,
                            write_iterator, counters_dict, is_mpi)
    
    
    ######## Build close section ########
    close_section = close_build(nthreads, files_dict, metas_dict, i_symbol, nthreads_dim, dswap_compression)
    
    
    iet_body.insert(0, pre_time_io_section)
    
    #TODO: Generate blank lines between sections
    for size_init in func_sizes_dict.values():
        iet_body.insert(0, size_init)
        
    if dswap_compression:
        type_eq = IREq(type_var, String(r"zfp_type_float"))
        c_type_eq = ClusterizedEq(type_eq, ispace=None)
        type_eq = Expression(c_type_eq, None, True)
        iet_body.insert(0, type_eq)
           
    iet_body.insert(0, float_size_init)
    iet_body.insert(0, open_section)
    iet_body.append(close_section)
    
    ######## Free slices memory ########
    if dswap_compression and not is_write:
        close_slices = close_slices_build(nthreads, i_symbol, slices_size_dict, nthreads_dim)
        iet_body.append(close_slices)
        
        for func in slices_size_dict:
            iet_body.append(Call(name="free", arguments=[(String(f"{func}_slices_size"))]))
        
    return iet_body


def open_build(files_array_dict, counters_array_dict, metas_dict, spt_dict, offset_dict, slices_size_dict, nthreads_dim, nthreads, is_write, i_symbol, dswap_compression):
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
        dswap_compression (CompressionConfig): object representing compression settings

    Returns:
        Section: open section
    """
    
    # Build conditional
    # Regular Forward or Gradient
    arrays = [file_array for file_array in files_array_dict.values()] 
    if not dswap_compression and not is_write:
        arrays.extend(counters_array for counters_array in counters_array_dict.values())
    # Compression Forward or Compression Gradient
    if dswap_compression:
        arrays.extend(metas_array for metas_array in metas_dict.values())
    if dswap_compression and not is_write:
        arrays.extend(spt_array for spt_array in spt_dict.values())
        arrays.extend(offset_array for offset_array in offset_dict.values()) 

    arrays_cond = dswap_array_alloc_check(arrays) 
    
    #Call open_thread_files
    open_threads_calls = []
    for func_name in files_array_dict:
        func_args = [files_array_dict[func_name], nthreads, String('"{}"'.format(func_name))]
        if dswap_compression:
            func_args.append(metas_dict[func_name])
        open_threads_calls.append(Call(name='open_thread_files_temp', arguments=func_args))

    # Open section body
    body = [arrays_cond, *open_threads_calls]
    
    # Additional initialization for Gradient operators
    if not is_write and not dswap_compression:
        # Regular
        counters_init = []
        interval_group = IntervalGroup((Interval(nthreads_dim, 0, nthreads)))
        for counter in counters_array_dict.values():
            counters_eq = ClusterizedEq(IREq(counter[i_symbol], 1), ispace=IterationSpace(interval_group))
            counters_init.append(Expression(counters_eq, None, False))
        
        open_iteration_grad = Iteration(counters_init, nthreads_dim, nthreads-1)
        body.append(open_iteration_grad)
    
    elif not is_write and dswap_compression:
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
        
    return Section("open", body, time_only_profiling=True)


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


def init_offload_build(funcs_dict, nthreads, files_dict, func_sizes_symb_dict, is_mpi, metas_dict, dswap_compression, type_var):
    """
    Builds the section responsable for offloading the initial timesteps to disk

    Args:
        funcs_dict (Dictonary): dict with Functions defined by user for the operator
        nthreads (NThreads): number of threads
        files_dict (Dictonary): dict with arrays of files
        func_sizes_symb_dict (Dictonary): dict with Function sizes symbols
        is_mpi (bool): MPI execution flag
        metas_dict (Dictonary): dict with arrays of metadata
        dswap_compression (CompressionConfig): object with compression settings
        type_var (Symbol): representation of zfp_type_float

    Returns:
        init_offload_section: Section offloading initial timesteps
    """
    to_inits = []
    iterations = []
    for func in funcs_dict:
        time_order = Symbol(name=func+"_time_order", dtype=np.int16)
        time_dim_size = funcs_dict[func].symbolic_shape[0]
        time_order_eq = IREq(time_order, time_dim_size - 1)
        time_order_exp = Expression(ClusterizedEq(time_order_eq, ispace=None), init=True)
        to_inits.append(time_order_exp)
        
        time = Symbol(name="time", dtype=np.int32, ignoreDefinition=True)
        if dswap_compression:
            # Create iteration space
            func_size1 = funcs_dict[func].symbolic_shape[1]
            func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
            interval = Interval(func_size_dim, 0, func_size1)
            interval_group = IntervalGroup((interval))
            ispace = IterationSpace(interval_group)    
            
            #Initialize tid
            i_symbol = Symbol(name="i", dtype=np.int32)
            tid = Symbol(name="tid", dtype=np.int32)
            c_tid_eq = ClusterizedEq(IREq(tid, Mod(i_symbol, nthreads)), ispace=ispace)
            
            pragma = Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
            write_iteration = compress_build(files_dict[func], metas_dict[func], funcs_dict[func], i_symbol, pragma,
                                        func_size_dim, tid, c_tid_eq, time, dswap_compression, type_var)
        else:
            write_iteration = write_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], time, is_mpi)
        
        time_dim = CustomDimension(name="time", symbolic_size=time_dim_size)
        time_iteration = Iteration(write_iteration, time_dim, time_order-1)
        iterations.append(time_iteration)
    
    return Section("offload", to_inits + iterations, time_only_profiling=True)
        
def init_offset_build(funcs_dict, nthreads, counters_dict, dswap_compression, spt_dict, offset_dict, slices_size_dict):
    """
    Builds the section responsable for performing the initial timesteps displacement
    from disk

    Args:
        funcs_dict (Dictonary): dict with Functions defined by user for the operator
        nthreads (NThreads): number of threads
        counters_dict (Dictonary): dict with counter arrays for each Function
        dswap_compression (CompressionConfig): object with compression settings
        spt_dict (Array): dict with arrays of slices per thread
        offset_dict (Array): dict with arrays of offset
        slices_size_dict (PointerArray): dict with 2d-arrays of slices for compression mode

    Returns:
        init_offset_section: Section displacing initial timesteps
    """
    to_inits = []
    iterations = []
    for func in funcs_dict:   
        # Time order expression
        time_order = Symbol(name=func+"_time_order", dtype=np.int16)
        time_dim_size = funcs_dict[func].symbolic_shape[0]
        time_order_eq = IREq(time_order, time_dim_size - 1)
        time_order_exp = Expression(ClusterizedEq(time_order_eq, ispace=None), init=True)
        to_inits.append(time_order_exp)
        
        # Iteration space creation
        func_size1 = funcs_dict[func].symbolic_shape[1]
        i_dim = CustomDimension(name="i", symbolic_size=func_size1)
        interval = Interval(i_dim, 0, func_size1)
        interval_group = IntervalGroup((interval))
        ispace = IterationSpace(interval_group)

        it_nodes=[]
        # Initialize tid
        tid = Symbol(name="tid", dtype=np.int32)
        tid_eq = IREq(tid, Mod(i_dim, nthreads))
        c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
        it_nodes.append(Expression(c_tid_eq, None, True))
        
        if dswap_compression:
            #Get slice 
            slice = Symbol(name="slice", dtype=np.int32)
            spt = spt_dict[func]
            c_slice_eq = ClusterizedEq(IREq(slice, spt[tid]), ispace=ispace)
            it_nodes.append(Expression(c_slice_eq, None, True))
            
            #Offset increment
            offset = offset_dict[func]
            slice_size = slices_size_dict[func]
            c_offset_eq = ClusterizedEq(IREq(offset[tid], slice_size[tid, slice]), ispace=ispace)
            it_nodes.append(Increment(c_offset_eq))
            
            #Spt increment
            c_spt_eq = ClusterizedEq(IREq(spt[tid], -1), ispace=ispace)
            it_nodes.append(Increment(c_spt_eq))
        else:
            # Counters increment
            counters = counters_dict[func]
            counters_inc = IREq(counters[tid], 1)
            c_new_counters_eq = ClusterizedEq(counters_inc, ispace=ispace)
            it_nodes.append(Increment(c_new_counters_eq))
        
        #Read iteration
        pragma = Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
        read_iteration = Iteration(it_nodes, i_dim, func_size1-1, direction=Backward, pragmas=[pragma])
        
        #Time iteration
        time_dim = CustomDimension(name="time", symbolic_size=time_dim_size)
        time_iteration = Iteration(read_iteration, time_dim, time_order-2)
        iterations.append(time_iteration)
        
        
    return Section("offset", to_inits + iterations, time_only_profiling=True) 

def compress_or_decompress_build(files_dict, metas_dict, iet_body, is_write, funcs_dict, nthreads, write_iterator,
                                 spt_dict, offset_dict, dswap_compression, slices_dict, type_var):
    """
    This function decides if it is either a compression or a decompression

    Args:
        files_dict (Dictonary): dict with arrays of files
        metas_dict (Dictonary): dict with arrays of metadata
        iet_body (List): IET body nodes
        is_write (bool): if True, it is write. It is read otherwise
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        nthreads (NThreads): number of threads
        write_iterator (ModuloDimension): time iterator in which the current timestep is
                                          computed and written in disk
        spt_dict (Array): dict with arrays of slices per thread
        offset_dict (Array): dict with arrays of offset
        dswap_compression (CompressionConfig): object with compression settings
        slices_dict (PointerArray): dict with 2d-arrays of slices for compression mode
        type_var (Symbol): representation of zfp_type_float
    """
    
    sec_name = "compress" if is_write else "decompress"
    pragma = Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    expressions = FindNodes(Expression).visit(iet_body)
    iterations=[]
    for func in funcs_dict:
        # Create iteration space
        func_size1 = funcs_dict[func].symbolic_shape[1]
        func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
        interval = Interval(func_size_dim, 0, func_size1)
        interval_group = IntervalGroup((interval))
        ispace = IterationSpace(interval_group)    
        
        #Initialize tid
        i_symbol = Symbol(name="i", dtype=np.int32)
        tid = Symbol(name="tid", dtype=np.int32)
        c_tid_eq = ClusterizedEq(IREq(tid, Mod(i_symbol, nthreads)), ispace=ispace)
        
        if is_write:
            io_iteration = compress_build(files_dict[func], metas_dict[func], funcs_dict[func], i_symbol, pragma,
                                         func_size_dim, tid, c_tid_eq, write_iterator, dswap_compression, type_var)
        else:
            time_iter = dswap_get_read_time_iterator(expressions, funcs_dict[func])
            io_iteration = decompress_build(files_dict[func], funcs_dict[func], i_symbol, pragma, func_size_dim, tid, c_tid_eq,
                                ispace, time_iter, spt_dict[func], offset_dict[func], dswap_compression, slices_dict[func], type_var)
        
        iterations.append(io_iteration)
    
    io_section = Section(sec_name, iterations, time_only_profiling=True)

    dswap_update_iet(iet_body, sec_name + "_temp", io_section)      
    

def compress_build(files_array, metas_array, func_stencil, i_symbol, pragma, func_size_dim, tid,
                   c_tid_eq, write_iterator, dswap_compression, type_var):
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
        write_iterator (ModuloDimension): time iterator in which the current timestep is
                                          computed and written in disk
        dswap_compression (CompressionConfig): object with compression settings
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
    
    arguments_field = [func_stencil[write_iterator,i_symbol], type_var, func_stencil.symbolic_shape[-1]]
    if len(func_stencil.symbolic_shape) == 4:
        dim = 2
        arguments_field.insert(2, func_stencil.symbolic_shape[-2])
    else:
        dim = 1
    it_nodes.append(Call(name=f"zfp_field_{dim}d", arguments=arguments_field, retobj=field))
    it_nodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    it_nodes.append(dswap_get_compress_mode_function(dswap_compression, zfp, field, type_var))
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

def decompress_build(files_array, func_stencil, i_symbol, pragma, func_size_dim, tid, c_tid_eq, ispace, t2, spt_array, offset_array, dswap_compression, slices_size, type_var):
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
        dswap_compression (CompressionConfig): object with compression settings
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
    
    arguments_field = [func_stencil[t2,i_symbol], type_var, func_stencil.symbolic_shape[-1]]
    if len(func_stencil.symbolic_shape) == 4:
        dim = 2
        arguments_field.insert(2, func_stencil.symbolic_shape[-2])
    else:
        dim = 1
    it_nodes.append(Call(name=f"zfp_field_{dim}d", arguments=arguments_field, retobj=field))
    it_nodes.append(Call(name="zfp_stream_open", arguments=[Null], retobj=zfp))
    it_nodes.append(dswap_get_compress_mode_function(dswap_compression, zfp, field, type_var))
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

def write_or_read_build(iet_body, is_write, nthreads, files_dict, func_sizes_symb_dict, funcs_dict, write_iterator, counters_dict, is_mpi):
    """
    Builds operator's read or write section, depending on the disk_swap mode.
    Replaces the temporary section at the end of the time iteration by the read or write section.   

    Args:
        iet_body (List): list of IET nodes 
        is_write (bool): True for the Forward operator; False for the Gradient operator
        nthreads (NThreads): symbol of number of threads
        files_dict (Dictonary): dict with arrays of files
        func_sizes_symb_dict (Dictonary): dict with Function sizes symbols
        funcs_dict (Dictonary): dict with Functions defined by user for Operator
        write_iterator (ModuloDimension): time iterator in which the current timestep is
                                          computed and written in disk
        counters_dict (Dictonary): dict with counter arrays for each Function
        is_mpi (bool): MPI execution flag

    """
    io_body=[]
    if is_write:
        temp_name = "write_temp"
        name = "write"
        for func in funcs_dict:
            func_write = write_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func], write_iterator, is_mpi)
            io_body.append(func_write)
        
    else: # read
        temp_name = "read_temp"
        name = "read"
        expressions = FindNodes(Expression).visit(iet_body)
        for func in funcs_dict:
            func_read = read_build(nthreads, files_dict[func], func_sizes_symb_dict[func], funcs_dict[func],
                                   counters_dict[func], expressions)
            io_body.append(func_read)
          
    io_section = Section(name, io_body, time_only_profiling=True)
    dswap_update_iet(iet_body, temp_name, io_section)     


def write_build(nthreads, files_array, func_size, func_stencil, write_iterator, is_mpi):
    """
    Builds write iteration of given Function   

    Args:
        nthreads (NThreads): symbol of number of threads
        files_array (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        func_size (Symbol): the func_stencil size
        func_stencil (u): a stencil we call u
        write_iterator (ModuloDimension): time iterator in which the current timestep is
                                         computed and written in disk
        is_mpi (bool): MPI execution flag

    Returns:
        Iteration: write loop
    """

    # Create iteration space
    func_size1 = func_stencil.symbolic_shape[1]
    func_size_dim = CustomDimension(name="i", symbolic_size=func_size1)
    interval = Interval(func_size_dim, 0, func_size1)
    interval_group = IntervalGroup((interval))
    ispace = IterationSpace(interval_group)
    it_nodes = []

    # Initialize tid
    tid = Symbol(name="tid", dtype=np.int32)
    tid_eq = IREq(tid, Mod(func_size_dim, nthreads))
    c_tid_eq = ClusterizedEq(tid_eq, ispace=ispace)
    it_nodes.append(Expression(c_tid_eq, None, True))
    
    # Write call
    ret = Symbol(name="ret", dtype=np.int32)
    write_call = Call(name="write", arguments=[files_array[tid], func_stencil[write_iterator, func_size_dim], func_size], retobj=ret)
    it_nodes.append(write_call)
    
    # Error conditional
    pstring = String("\"Write size mismatch with function slice size\"")
    cond_nodes = [Call(name="perror", arguments=pstring)]
    cond_nodes.append(Call(name="exit", arguments=1))
    cond = Conditional(CondNe(ret, func_size), cond_nodes)
    it_nodes.append(cond)

    # TODO: Pragmas should depend on the user's selected optimization options and be generated by the compiler
    if is_mpi:
        pragma = Pragma("omp parallel for schedule(static,1)")
    else:
        pragma = Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")

    return Iteration(it_nodes, func_size_dim, func_size1-1, pragmas=[pragma])

def read_build(nthreads, files_array, func_size, func_stencil, counters, expressions):
    """
    Builds read iteration of given Function  

    Args:
        nthreads (NThreads): symbol of number of threads
        files_array (files): pointer of allocated memory of nthreads dimension. Each place has a size of int
        func_size (Symbol): the func_stencil size
        func_stencil (u): a stencil we call u
        counters (array): pointer of allocated memory of nthreads dimension. Each place has a size of int
        expressions(array): list of all expressions in iet_body
    Returns:
        Iteration: read loop
    """

    time_iter = dswap_get_read_time_iterator(expressions, func_stencil)
    pragma = Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    
    # Create iteration space
    func_size1 = func_stencil.symbolic_shape[1]
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

    # Read call
    ret = Symbol(name="ret", dtype=np.int32)
    read_call = Call(name="read", arguments=[files_array[tid], func_stencil[time_iter, i_dim], func_size], retobj=ret)
    it_nodes.append(read_call)

    # Error conditional
    pstring = String("\"Cannot open output file\"")
    cond_nodes = [
        Call(name="printf", arguments=[String("\"%d\""), ret]),
        Call(name="perror", arguments=pstring), 
        Call(name="exit", arguments=1)
    ]
    cond = Conditional(CondNe(ret, func_size), cond_nodes) # if (ret != func_size)
    it_nodes.append(cond)
    
    # Counters increment
    counters_inc = IREq(counters[tid], 1)
    c_new_counters_eq = ClusterizedEq(counters_inc, ispace=ispace)
    it_nodes.append(Increment(c_new_counters_eq))
        
    return Iteration(it_nodes, i_dim, func_size1-1, direction=Backward, pragmas=[pragma])


def close_build(nthreads, files_dict, metas_dict, i_symbol, nthreads_dim, dswap_compression):
    """
    This method inteds to ls read.c close section.
    Obs: maybe the desciption of the variables should be better

    Args:
        nthreads (NThreads): symbol of number of threads
        files_dict (dict): dictionary with file pointers arrays
        metas_dict (dict): dictionary with meta file pointers arrays
        i_symbol (Symbol): symbol of the iterator index i
        nthreads_dim (CustomDimension): dimension from 0 to nthreads
        dswap_compression (CompressionConfig): object representing compression settings

    Returns:
        Section: complete close section
    """

    it_nodes=[]
    
    for func in files_dict:
        files = files_dict[func]
        it_nodes.append(Call(name="close", arguments=[files[i_symbol]]))
        
    if dswap_compression:
        for func in metas_dict:
            metas = metas_dict[func]
            it_nodes.append(Call(name="close", arguments=[metas[i_symbol]])) 
    
    close_iteration = Iteration(it_nodes, nthreads_dim, nthreads-1)
    
    return Section("close", close_iteration, time_only_profiling=True)


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