import numpy as np
from sympy import Mod
from pdb import set_trace

from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, Macro, String, cast_mapper, SizeOf)
from devito.symbolics.extended_sympy import (FieldFromPointer, Byref)
from devito.types import CustomDimension, Array, Symbol, Pointer, FILE, Timer, NThreads, off_t, size_t, PointerArray
from devito.ir.iet import (Expression, Iteration, Conditional, Call, Conditional, CallableBody, Callable,
                            FindNodes, Transformer, Return, Definition)
from devito.ir.equations import IREq, ClusterizedEq

__all__ = ['ooc_efuncs']


def open_threads_build(nthreads, filesArray, metasArray, iSymbol, nthreadsDim, nameArray, is_forward, is_mpi, is_compression):
    """
    This method generates the function open_thread_files according to the operator used.

    Args:
        nthreads (NThreads): number of threads
        filesArray (Array): array of files
        metasArray (Array): some array
        nthreadsDim (CustomDimension): dimension i from 0 to nthreads
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        is_mpi (bool): True for the use of MPI; False otherwise.
        is_compression (bool): True for the use of compression; False otherwise.

    Returns:
        Callable: the callable function open_thread_files
    """
    
    itNodes=[]
    ifNodes=[]
    
    # TODO: initialize char name[100]
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32, ignoreDefinition=True)

    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    stencilNameArray = Array(name='stencil', dimensions=nameDim, dtype=np.byte)        
    
    ifNodes.append(Call(name="perror", arguments=String("\"Cannot open output file\\n\"")))
    ifNodes.append(Call(name="exit", arguments=1))
       
    if is_mpi:
        # TODO: initialize int myrank
        # TODO: initialize char error[140]
        myrank = Symbol(name="myrank", dtype=np.int32)
        mrEq = IREq(myrank, 0)

        dps = Symbol(name="DPS", dtype=np.int32, ignoreDefinition=True)
        socket = Symbol(name="socket", dtype=np.int32)
        
        nvmeIdEq = IREq(nvme_id, Mod(iSymbol, ndisks)+socket)        
        socketEq = IREq(socket, Mod(myrank, 2) * dps)
        cSocketEq = ClusterizedEq(socketEq, ispace=None)
        cNvmeIdEq = ClusterizedEq(nvmeIdEq, ispace=None)                  
        
        # TODO: MPI_COMM_WORLD as Macro. Try to find a berrer IR to &myrank 
        itNodes.append(Expression(ClusterizedEq(mrEq), None, True))
        itNodes.append(Call(name="MPI_Comm_rank", arguments=[Macro("MPI_COMM_WORLD"), Byref(myrank)]))
        itNodes.append(Expression(cSocketEq, None, True)) 
        itNodes.append(Expression(cNvmeIdEq, None, True)) 
        itNodes.append(Call(name="sprintf", arguments=[nameArray, String("\"data/nvme%d/socket_%d_thread_%d.data\""), nvme_id, myrank, iSymbol]))
    else:
        nvmeIdEq = IREq(nvme_id, Mod(iSymbol, ndisks))
        cNvmeIdEq = ClusterizedEq(nvmeIdEq, ispace=None)        
        itNodes.append(Expression(cNvmeIdEq, None, True))   
        itNodes.append(Call(name="sprintf", arguments=[nameArray, String("\"data/nvme%d/%s_vec_%d.bin\""), nvme_id, stencilNameArray, iSymbol]))        
    
    opFlagsStr = String("OPEN_FLAGS")
    opFlagsStrCompFwd = String("O_WRONLY | O_CREAT | O_TRUNC")
    opFlagsStrCompGrd = String("O_RDONLY")
    flagsStr = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")        
    
    if is_forward and is_compression:
        itNodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), nameArray]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStrCompFwd, flagsStr], retobj=filesArray[iSymbol]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStrCompFwd, flagsStr], retobj=metasArray[iSymbol]))
    elif is_forward and not is_compression:
        itNodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), nameArray]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStr, flagsStr], retobj=filesArray[iSymbol]))
    elif not is_forward and is_compression:
        itNodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), nameArray]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStrCompGrd, flagsStr], retobj=filesArray[iSymbol]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStrCompGrd, flagsStr], retobj=metasArray[iSymbol]))
    elif not is_forward and not is_compression:
        itNodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), nameArray]))
        itNodes.append(Call(name="open", arguments=[nameArray, opFlagsStr, flagsStr], retobj=filesArray[iSymbol]))   
    
    itNodes.append(Conditional(CondEq(filesArray[iSymbol], -1), ifNodes))
    funcArgs = [filesArray, nthreads, stencilNameArray]
    if is_compression:
        itNodes.append(Call(name="sprintf", arguments=[nameArray, String("\"data/nvme%d/thread_%d.data\""), nvme_id, iSymbol]))
        if is_forward:
            itNodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), nameArray]))
        else:
            itNodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), nameArray]))
        itNodes.append(Conditional(CondEq(metasArray[iSymbol], -1), ifNodes))
        funcArgs = funcArgs.append(metasArray)
    
    openIteration = Iteration(itNodes, nthreadsDim, nthreads)
    
    body = CallableBody(openIteration)
    callable = Callable("open_thread_files", body, "void", funcArgs)

    return callable

def get_slices_build(sptArray, nthreads, metasArray, nthreadsDim, iSymbol):
    """_summary_

    Args:
        sptArray (_type_): _description_
        nthreads (_type_): _description_
        metasArray (_type_): _description_
        nthreadsDim (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    itNodes=[]
    ifNodes=[]
    funcBody=[]
    
    slicesSize = PointerArray(name='slices_size', dimensions=[nthreadsDim], array=Array(name='slices_size', dimensions=[nthreadsDim], dtype=size_t))
    mAllocCall = Call(name="(size_t**) malloc", arguments=[nthreads*SizeOf(Pointer(name='size_t *', dtype=size_t))], retobj=slicesSize)
    funcBody.append(mAllocCall)
    
    # Get size of the file
    fSize = Symbol(name='fsize', dtype=off_t)
    lseekCall = Call(name="lseek", arguments=[metasArray[iSymbol], cast_mapper[size_t](0), Macro("SEEK_END")], retobj=fSize)
    itNodes.append(lseekCall)
    
    # Get number of slices per thread file
    sptEq = IREq(sptArray[iSymbol], cast_mapper[int](fSize) / SizeOf(Symbol(name='size_t', dtype=size_t)) -1)
    cSptEq = ClusterizedEq(sptEq, ispace=None)
    itNodes.append(Expression(cSptEq, None, False))
    
    # Allocate
    slicesSizeTidMallocCall = Call(name='malloc', arguments=[fSize], retobj=slicesSize[iSymbol])
    slicesSizeTidCast = cast_mapper[(size_t, '*')](slicesSize[iSymbol])
    slicesSizeTidCastEq = IREq(slicesSize[iSymbol], slicesSizeTidCast)
    cSlicesSizeTidCastEq = ClusterizedEq(slicesSizeTidCastEq, ispace=None)
    itNodes.append(slicesSizeTidMallocCall)
    itNodes.append(Expression(cSlicesSizeTidCastEq, None, False))
    
    ifNodes.append(Call(name="perror", arguments=String("\"Error to allocate slices\\n\"")))
    ifNodes.append(Call(name="exit", arguments=1))
    itNodes.append(Conditional(CondEq(slicesSize, Macro("NULL")), ifNodes))
    
    # Return to begin of the file
    itNodes.append(Call(name="lseek", arguments=[metasArray[iSymbol], 0, Macro("SEEK_SET")]))
    
    # Read to slices_size buffer
    itNodes.append(Call(name="read", arguments=[metasArray[iSymbol], Byref(slicesSize[iSymbol, 0]), fSize]))
    
    getSlicesIteration = Iteration(itNodes, nthreadsDim, nthreads-1)
    funcBody.append(getSlicesIteration)
    funcBody.append(Return(String(r"slices_size")))
        
    getSliceSizeBody = CallableBody(funcBody)
    callable = Callable("get_slices_size", getSliceSizeBody, "size_t**", [metasArray, sptArray, nthreads])
    return callable    
    

@iet_pass
def ooc_efuncs(iet, **kwargs):
    """_summary_

    Args:
        iet (_type_): _description_

    Returns:
        _type_: _description_
    """
            
    is_forward = kwargs['options']['out-of-core'].mode == 'forward'
    is_mpi = kwargs['options']['mpi']
    is_compression = kwargs['options']['out-of-core'].compression
    efuncs = []
    mapper={}
    calls = FindNodes(Call).visit(iet)

    nthreads = NThreads(ignoreDefinition=True)
    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)
    iSymbol = Symbol(name="i", dtype=np.int32)

    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads) 
    filesArray = Array(name='files', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=True)
    metasArray = Array(name='metas', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=(not is_forward))

    if is_compression and not is_forward:
            sptArray = Array(name='spt', dimensions=[nthreadsDim], dtype=np.int32)
            slices_size = PointerArray(name='slices_size', dimensions=[nthreadsDim],
                                        array=Array(name='slices_size', dimensions=[nthreadsDim], dtype=size_t))
            new_get_slices_call = Call(name='get_slices_size', arguments=[metasArray, sptArray, nthreads], retobj=slices_size)
            slicesSizeCallable = get_slices_build(sptArray, nthreads, metasArray, nthreadsDim, iSymbol)
            efuncs.append(slicesSizeCallable)
            get_slices_call = next((call for call in calls if call.name == 'get_slices_size_temp'), None)
            mapper[get_slices_call] = new_get_slices_call

    openThreadsCallable = open_threads_build(nthreads, filesArray, metasArray,iSymbol,
                                             nthreadsDim, nameArray, is_forward, 
                                             is_mpi, is_compression)
    efuncs.append(openThreadsCallable)   
    for call in calls:
        if call.name == 'open_thread_files_temp':
            new_open_thread_call = Call(name='open_thread_files', arguments=call.arguments)
            mapper[call] = new_open_thread_call
    
    iet = Transformer(mapper).visit(iet)   
    
    return iet, {'efuncs': efuncs}


"""
funcArgs = [filesArray, nthreads]
    if is_compression: funcArgs.insert(1, metasArray)
    new_open_thread_call = Call(name='open_thread_files', arguments=funcArgs)

    openThreadsCallable = open_threads_build(nthreads, filesArray, metasArray,iSymbol,
                                             nthreadsDim, nameArray, is_forward, 
                                             is_mpi, is_compression)
    efuncs.append(openThreadsCallable)   
    open_threads_call = next((call for call in calls if call.name == 'open_thread_files_temp'), None)
    mapper[open_threads_call] = new_open_thread_call
"""