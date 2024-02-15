import numpy as np
from sympy import Mod

from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, Macro, String)
from devito.symbolics.extended_sympy import (FieldFromPointer, Byref)
from devito.types import CustomDimension, Array, Symbol, Pointer, FILE, Timer, NThreads
from devito.ir.iet import (Expression, Iteration, Conditional, Call, Conditional, CallableBody, Callable,
                            FindNodes, Transformer, Return)
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


def open_threads_build(nthreads, filesArray, iSymbol, nthreadsDim, nameArray, is_forward, is_mpi):
    """
    This method generates the function open_thread_files according to the operator used.

    Args:
        nthreads (NThreads): number of threads
        filesArray (Array): array of files
        iSymbol (Symbol): symbol of the iterator index i 
        nthreadsDim (CustomDimension): dimension i from 0 to nthreads
        is_forward (bool): True for the Forward operator; False for the Gradient operator
        is_mpi (bool): True for the use of MPI; False otherwise.

    Returns:
        Callable: the callable function open_thread_files
    """
    
    itNodes=[]
    ifNodes=[]
    
    # TODO: initialize char name[100]
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32, ignoreDefinition=True)

    opFlagsStr = String("OPEN_FLAGS")
    flagsStr = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")
    openCall = Call(name="open", arguments=[nameArray, opFlagsStr, flagsStr], retobj=filesArray[iSymbol])
    
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
        
    
    if is_forward:
        itNodes.append(Call(name="printf", arguments=[String("\"Creating file %s\\n\""), nameArray]))
    else:
        itNodes.append(Call(name="printf", arguments=[String("\"Reading file %s\\n\""), nameArray]))


    ifNodes.append(Call(name="perror", arguments=String("\"Cannot open output file\\n\"")))
    ifNodes.append(Call(name="exit", arguments=1))
    openCond = Conditional(CondEq(filesArray[iSymbol], -1), ifNodes) 
    
    itNodes.append(openCall)   
    
    itNodes.append(openCond)

    openIteration = Iteration(itNodes, nthreadsDim, nthreads-1)
    
    body = CallableBody(openIteration)
    callable = Callable("open_thread_files", body, "void", [filesArray, nthreads])

    return callable


@iet_pass
def ooc_efuncs(iet, **kwargs):
    is_forward = kwargs['options']['out-of-core'].mode == 'forward'
    is_mpi = kwargs['options']['mpi']
    profiler_name = kwargs['profiler'].name

    nthreads = NThreads(ignoreDefinition=True)
    timerProfiler = Timer(profiler_name, [], ignoreDefinition=True)
    size_name = 'write_size' if is_forward else 'read_size'
    io_size = Symbol(name=size_name, dtype=np.int64)
    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)

    # new_save_call = Call(name="save", arguments=[nthreads, timerProfiler, io_size])
    # saveCallable = save_build(nthreads, timerProfiler, io_size, nameArray, is_forward, is_mpi)

    nthreadsDim = CustomDimension(name="i", symbolic_size=nthreads) 
    filesArray = Array(name='files', dimensions=[nthreadsDim], dtype=np.int32, ignoreDefinition=True)
    iSymbol = Symbol(name="i", dtype=np.int32)

    new_open_thread_call = Call(name='open_thread_files', arguments=[filesArray, nthreads])
    openThreadsCallable = open_threads_build(nthreads, filesArray, iSymbol, nthreadsDim, nameArray, is_forward, is_mpi)

    calls = FindNodes(Call).visit(iet)
    # save_call = next((call for call in calls if call.name == 'save_temp'), None)
    open_threads_call = next((call for call in calls if call.name == 'open_thread_files_temp'), None)

    mapper={
        # save_call: new_save_call,
        open_threads_call: new_open_thread_call}
    iet = Transformer(mapper).visit(iet)
    efuncs=[
        # saveCallable, 
        openThreadsCallable]
    return iet, {'efuncs': efuncs}