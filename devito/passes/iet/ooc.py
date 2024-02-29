import numpy as np
from sympy import Mod
from pdb import set_trace

from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, Macro, String, cast_mapper, SizeOf, Null)
from devito.symbolics.extended_sympy import (FieldFromPointer, Byref)
from devito.types import CustomDimension, Array, Symbol, Pointer, FILE, Timer, NThreads, off_t, size_t, PointerArray
from devito.ir.iet import (Expression, Iteration, Conditional, Call, Conditional, CallableBody, Callable,
                            FindNodes, Transformer, Return, Definition)
from devito.ir.equations import IREq, ClusterizedEq

__all__ = ['ooc_efuncs']

def save_build(nthreads, timerProfiler, io_size, nameArray, is_forward, is_mpi):
    """
    This method generates the function save according to the operator used.

    Args:
        nthreads (Nthreads): number of threads
        timerProfiler (Timer): a Timer to represent a profiler
        io_size (Symbol): a symbol which represents write_size or read_size
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        is_mpi (bool): True for the use of MPI; False otherwise.

    Returns:
        Callable: the callable function save
    """

    funcNodes = []
    
    if is_mpi:
        if is_forward:
            opStrTitle = String("\">>>>>>>>>>>>>> MPI FORWARD <<<<<<<<<<<<<<<<<\\n\"")
            tagOp = "FWD"
            operation = "write"
        else:
            opStrTitle = String("\">>>>>>>>>>>>>> MPI REVERSE <<<<<<<<<<<<<<<<<\\n\"")
            tagOp = "REV"
            operation = "read"            
    else:
        if is_forward:
            opStrTitle = String("\">>>>>>>>>>>>>> FORWARD <<<<<<<<<<<<<<<<<\\n\"")
            tagOp = "FWD"
            operation = "write"
        else:
            opStrTitle = String("\">>>>>>>>>>>>>> REVERSE <<<<<<<<<<<<<<<<<\\n\"")
            tagOp = "REV"
            operation = "read"  
        

    if is_mpi:
        myrank = Symbol(name="myrank", dtype=np.int32)
        mrEq = IREq(myrank, 0)
        funcNodes.append(Expression(ClusterizedEq(mrEq), None, True))
        funcNodes.append(Call(name="MPI_Comm_rank", arguments=[Macro("MPI_COMM_WORLD"), Byref(myrank)]))
        funcNodes.append(Conditional(CondNe(myrank, 0), Return()))

    funcNodes.append(Call(name="printf", arguments=[opStrTitle]))
    
    pstring = String("\"Threads %d\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, nthreads]))

    pstring = String("\"Disks %d\\n\"")
    ndisksStr = String("NDISKS")
    funcNodes.append(Call(name="printf", arguments=[pstring, ndisksStr]))

    #TODO: Must retrieve section names from somewhere
    tSec0 = FieldFromPointer("section0", timerProfiler)
    tSec1 = FieldFromPointer("section1", timerProfiler)
    tSec2 = FieldFromPointer("section2", timerProfiler)
    tOpen = FieldFromPointer("open", timerProfiler)
    tOp = FieldFromPointer(operation, timerProfiler)
    tClose = FieldFromPointer("close", timerProfiler)    

    pstring = String(f"\"[{tagOp}] Section0 %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tSec0]))

    pstring = String(f"\"[{tagOp}] Section1 %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tSec1]))

    pstring = String(f"\"[{tagOp}] Section2 %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tSec2])) 

    pstring = String("\"[IO] Open %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tOpen]))

    pstring = String(f"\"[IO] {operation.title()} %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tOp]))

    pstring = String("\"[IO] Close %.2lf s\\n\"")
    funcNodes.append(Call(name="printf", arguments=[pstring, tClose]))
    

    fileOpenNodes = []
    pstring = String(f"\"{tagOp.lower()}_disks_%d_threads_%d.csv\"")
    fileOpenNodes.append(Call(name="sprintf", arguments=[nameArray, pstring, ndisksStr, nthreads]))

    pstring = String("\"w\"")
    filePointer = Pointer(name="fpt", dtype=FILE)
    fileOpenNodes.append(Call(name="fopen", arguments=[nameArray, pstring], retobj=filePointer))

    filePrintNodes = []
    pstring = String(f"\"Disks, Threads, Bytes, [{tagOp}] Section0, [{tagOp}] Section1, [{tagOp}] Section2, [IO] Open, [IO] {operation.title()}, [IO] Close\\n\"")
    filePrintNodes.append(Call(name="fprintf", arguments=[filePointer, pstring]))
    
    pstring = String("\"%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\\n\"")
    filePrintNodes.append(Call(name="fprintf", arguments=[filePointer, pstring, ndisksStr, nthreads, io_size,
                                                          tSec0, tSec1, tSec2, tOpen, tOp, tClose]))

    body = funcNodes+fileOpenNodes+filePrintNodes
    body.append(Call(name="fclose", arguments=filePointer))
    saveCallBody = CallableBody(body)
    saveCallable = Callable("save", saveCallBody, "void", [nthreads, timerProfiler, io_size])

    return saveCallable


def open_threads_build(nthreads, filesArray, metasArray, nthreadsDim, nameArray, is_forward, is_mpi, is_compression):
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
    iSymbol = Symbol(name="i", dtype=np.int32)
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32, ignoreDefinition=True)        
    
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
        itNodes.append(Call(name="sprintf", arguments=[nameArray, String("\"data/nvme%d/thread_%d.data\""), nvme_id, iSymbol]))        
    
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
    funcArgs = [filesArray, nthreads]
    if is_compression:
        itNodes.append(Call(name="sprintf", arguments=[nameArray, String("\"data/nvme%d/thread_%d.data\""), nvme_id, iSymbol]))
        if is_forward:
            itNodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), nameArray]))
        else:
            itNodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), nameArray]))
        itNodes.append(Conditional(CondEq(metasArray[iSymbol], -1), ifNodes))
        funcArgs = [filesArray, metasArray, nthreads]
    
    openIteration = Iteration(itNodes, nthreadsDim, nthreads)
    
    body = CallableBody(openIteration)
    callable = Callable("open_thread_files", body, "void", funcArgs)

    return callable

def get_slices_build(sptArray, nthreads, metasArray, nthreadsDim):
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
    nthreadsDim.name='tid'
    
    slicesSize = PointerArray(name='slices_size', dimensions=[nthreadsDim], array=Array(name='slices_size', dimensions=[nthreadsDim], dtype=size_t))
    mAllocCall = Call(name="(size_t**) malloc", arguments=[nthreads*SizeOf(Pointer(name='size_t *', dtype=size_t))], retobj=slicesSize)
    funcBody.append(mAllocCall)
    
    # Get size of the file
    tid = Symbol(name="tid", dtype=np.int32)
    fSize = Symbol(name='fsize', dtype=off_t)
    lseekCall = Call(name="lseek", arguments=[metasArray[tid], cast_mapper[size_t](0), Macro("SEEK_END")], retobj=fSize)
    itNodes.append(lseekCall)
    
    # Get number of slices per thread file
    sptEq = IREq(sptArray[tid], cast_mapper[int](fSize) / SizeOf(String(r"size_t")) -1)
    cSptEq = ClusterizedEq(sptEq, ispace=None)
    itNodes.append(Expression(cSptEq, None, False))
    
    # Allocate
    slicesSizeTidMallocCall = Call(name='malloc', arguments=[fSize], retobj=slicesSize[tid])
    slicesSizeTidCast = cast_mapper[(size_t, '*')](slicesSize[tid])
    slicesSizeTidCastEq = IREq(slicesSize[tid], slicesSizeTidCast)
    cSlicesSizeTidCastEq = ClusterizedEq(slicesSizeTidCastEq, ispace=None)
    itNodes.append(slicesSizeTidMallocCall)
    itNodes.append(Expression(cSlicesSizeTidCastEq, None, False))
    
    ifNodes.append(Call(name="perror", arguments=String("\"Error to allocate slices\\n\"")))
    ifNodes.append(Call(name="exit", arguments=1))
    itNodes.append(Conditional(CondEq(slicesSize, Null), ifNodes))
    
    # Return to begin of the file
    itNodes.append(Call(name="lseek", arguments=[metasArray[tid], 0, Macro("SEEK_SET")]))
    
    # Read to slices_size buffer
    itNodes.append(Call(name="read", arguments=[metasArray[tid], Byref(slicesSize[tid, 0]), fSize]))
    
    getSlicesIteration = Iteration(itNodes, nthreadsDim, nthreads-1)
    funcBody.append(getSlicesIteration)
    funcBody.append(Return(String(r"slices_size")))
        
    getSliceSizeBody = CallableBody(funcBody)
    callable = Callable("get_slices_size", getSliceSizeBody, "size_t**", [metasArray, sptArray, nthreads])
    set_trace()
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
    profiler_name = kwargs['profiler'].name
    efuncs = []
    mapper={}
    calls = FindNodes(Call).visit(iet)

    nthreads = NThreads(ignoreDefinition=True)
    timerProfiler = Timer(profiler_name, [], ignoreDefinition=True)
    size_name = 'write_size' if is_forward else 'read_size'
    io_size = Symbol(name=size_name, dtype=np.int64)
    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)

    # new_save_call = Call(name="save", arguments=[nthreads, timerProfiler, io_size])
    # saveCallable = save_build(nthreads, timerProfiler, io_size, nameArray, is_forward, is_mpi)
    # efuncs.append(saveCallable)
    # save_call = next((call for call in calls if call.name == 'save_temp'), None)
    # mapper[save_call] = new_save_call

    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads) 
    filesArray = Array(name='files', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=True)
    if is_compression:
        metasArray = Array(name='metas', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=True)
        if not is_forward:
            sptArray = Array(name='spt', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=True)
            new_get_slices_call = Call(name='get_slices_size', arguments=[metasArray, sptArray, nthreads])
            slicesSizeCallable = get_slices_build(sptArray, nthreads, metasArray, nthreadsDim)
            efuncs.append(slicesSizeCallable)
            get_slices_call = next((call for call in calls if call.name == 'get_slices_size'), None)
            mapper[get_slices_call] = new_get_slices_call
            
    else:
        metasArray=None
        
    new_open_thread_call = Call(name='open_thread_files', arguments=[filesArray, nthreads])
    openThreadsCallable = open_threads_build(nthreads, filesArray, metasArray, 
                                             nthreadsDim, nameArray, is_forward, 
                                             is_mpi, is_compression)
    efuncs.append(openThreadsCallable)   
    open_threads_call = next((call for call in calls if call.name == 'open_thread_files'), None)
    mapper[open_threads_call] = new_open_thread_call
    
    iet = Transformer(mapper).visit(iet)   
    
    return iet, {'efuncs': efuncs}