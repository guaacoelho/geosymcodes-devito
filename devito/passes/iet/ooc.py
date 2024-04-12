import numpy as np
from sympy import Mod
from pdb import set_trace
from ctypes import c_int32, POINTER, c_int

from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, Macro, String, cast_mapper, SizeOf)
from devito.symbolics.extended_sympy import Byref
from devito.types import CustomDimension, Array, Symbol, Pointer, NThreads, off_t, size_t, PointerArray
from devito.ir.iet import (Expression, Iteration, Conditional, Call, Conditional, CallableBody, Callable,
                            FindNodes, Transformer, Return, Definition, EntryFunction)
from devito.ir.equations import IREq, ClusterizedEq

__all__ = ['ooc_efuncs']


def open_threads_build(nthreads, files_array, metas_array, i_symbol, nthreads_dim, name_array, is_forward, is_mpi, is_compression):
    """
    This method generates the function open_thread_files according to the operator used.

    Args:
        nthreads (NThreads): number of threads
        files_array (Array): array of files
        metas_array (Array): some array
        i_symbol (Symbol): iterator symbol
        nthreads_dim (CustomDimension): dimension i from 0 to nthreads
        name_array (Array): Function name
        is_forward (bool): True for the Forward operator; False for the Gradient operator
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
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String("\"data/nvme%d/socket_%d_thread_%d.data\""), nvme_id, myrank, i_symbol]))
    else:
        nvme_id_eq = IREq(nvme_id, Mod(i_symbol, ndisks))
        c_nvme_id_eq = ClusterizedEq(nvme_id_eq, ispace=None)        
        it_nodes.append(Expression(c_nvme_id_eq, None, True))   
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String("\"data/nvme%d/%s_vec_%d.bin\""), nvme_id, stencil_name_array, i_symbol]))        
    
    op_flags = String("OPEN_FLAGS")
    o_flags_comp_write = String("O_WRONLY | O_CREAT | O_TRUNC")
    o_flags_comp_read = String("O_RDONLY")
    s_flags = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")        
    
    if is_forward and is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_write, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String("\"data/nvme%d/thread_%d.data\""), nvme_id, i_symbol]))
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_write, s_flags], retobj=metas_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(metas_array[i_symbol], -1), if_nodes))
    elif is_forward and not is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, op_flags, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
    elif not is_forward and is_compression:
        it_nodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_read, s_flags], retobj=files_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(files_array[i_symbol], -1), if_nodes))
        it_nodes.append(Call(name="sprintf", arguments=[name_array, String("\"data/nvme%d/thread_%d.data\""), nvme_id, i_symbol]))
        it_nodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), name_array]))
        it_nodes.append(Call(name="open", arguments=[name_array, o_flags_comp_read, s_flags], retobj=metas_array[i_symbol]))
        it_nodes.append(Conditional(CondEq(metas_array[i_symbol], -1), if_nodes))
    elif not is_forward and not is_compression:
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
    


def headers_build(is_forward, is_compression, is_mpi):
    """
    Builds operator's headers

    Args:
        is_forward (bool): True for the write mode; False for read mode
        is_compression (bool): True for the compression operator; False otherwise
        is_mpi (bool): True for MPI execution; False otherwise

    Returns:
        headers (List) : list with header defines
        includes (List): list with includes
    """
    _out_of_core_mpi_headers=[(("ifndef", "DPS"), ("DPS", "4"))]
    _out_of_core_headers_forward=[("_GNU_SOURCE", ""),
                                  (("ifndef", "NDISKS"), ("NDISKS", "8")), 
                                  (("ifdef", "CACHE"), ("OPEN_FLAGS", "O_WRONLY | O_CREAT"), ("else", ), ("OPEN_FLAGS", "O_DIRECT | O_WRONLY | O_CREAT"))]
    _out_of_core_headers_gradient=[("_GNU_SOURCE", ""),
                                   (("ifndef", "NDISKS"), ("NDISKS", "8")), 
                                   (("ifdef", "CACHE"), ("OPEN_FLAGS", "O_RDONLY"), ("else", ), ("OPEN_FLAGS", "O_DIRECT | O_RDONLY"))]
    _out_of_core_compression_headers=[(("ifndef", "NDISKS"), ("NDISKS", "8")),]
    _out_of_core_includes = ["fcntl.h", "stdio.h", "unistd.h"]
    _out_of_core_mpi_includes = ["mpi.h"]
    _out_of_core_compression_includes = ["zfp.h"]


    # Headers
    headers=[]
    if is_compression:
        headers.extend(_out_of_core_compression_headers)
    else: 
        if is_forward:
            headers.extend(_out_of_core_headers_forward)
        else:
            headers.extend(_out_of_core_headers_gradient)

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
           
    is_forward = kwargs['options']['out-of-core'].mode == 'forward'
    is_mpi = kwargs['options']['mpi']
    is_compression = kwargs['options']['out-of-core'].compression
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

    if is_compression and not is_forward:
        spt_array = Array(name='spt', dimensions=[nthreads_dim], dtype=np.int32)
        slices_size = PointerArray(name='slices_size', dimensions=[nthreads_dim], array=Array(name='slices_size', dimensions=[nthreads_dim], dtype=size_t), ignoreDefinition=True)
        slices_size_callable = get_slices_build(spt_array, nthreads, metas_array, nthreads_dim, i_symbol, slices_size)
        efuncs.append(slices_size_callable)
        new_get_slices_call = Call(name='get_slices_size', arguments=[String(r"metas_vec"), String(r"spt_vec"), nthreads], 
                                   retobj=Pointer(name='slices_size', dtype=POINTER(POINTER(size_t)), ignoreDefinition=True))
        get_slices_call = next((call for call in calls if call.name == 'get_slices_size_temp'), None)
        mapper[get_slices_call] = new_get_slices_call

    open_threads_callable = open_threads_build(nthreads, files_array, metas_array, i_symbol,
                                             nthreads_dim, name_array, is_forward, 
                                             is_mpi, is_compression)
    efuncs.append(open_threads_callable)   
    for call in calls:
        if call.name == 'open_thread_files_temp':
            new_open_thread_call = Call(name='open_thread_files', arguments=call.arguments)
            mapper[call] = new_open_thread_call
    
    iet = Transformer(mapper).visit(iet)   
    
    headers, includes = headers_build(is_forward, is_compression, is_mpi)
    return iet, {'efuncs': efuncs, "headers": headers, "includes": includes}
