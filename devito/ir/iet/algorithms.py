import numpy as np

import ctypes as ct

import cgen

from sympy import Mod

from functools import reduce

from collections import OrderedDict

from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Call, Conditional, CallableBody, Callable, Return, FindSymbols)
from devito.ir.equations import IREq, ClusterizedEq
from devito.symbolics.extended_sympy import FieldFromPointer
from devito.tools import timed_pass
from devito.symbolics import (CondEq, CondNe, Macro, String)
from devito.types import CustomDimension, Array, PointerArray, Symbol, IndexedData, Pointer, FILE, Timer, NThreads
from devito.ir.support import (Interval, IntervalGroup, IterationSpace)

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree, **kwargs):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """

    out_of_core = kwargs['options']['out-of-core']
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            iet_body = queues.pop(i)
            if(out_of_core):
                iet_body = _ooc_build(iet_body, kwargs['sregistry'].nthreads, kwargs['profiler'])
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
            body = Iteration(queues.pop(i), i.dim, i.limits, direction=i.direction,
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
def _ooc_build(iet_body, nt, profiler):
    # Creates nthreads parameter representation.
    # It needs to be created once again in order to enable the ignoreDefinition flag,
    # avoinding multi definition of nthreads variable.
    nthreads = NThreads(ignoreDefinition=True)

    # Build files array
    cdim = [CustomDimension(name="nthreads", symbolic_size=nthreads)]
    filesArray = Array(name='files', dimensions=cdim, dtype=np.int32)

    # Test files array 
    pstring = String("'Error to alloc'")
    printfCall = Call(name="printf", arguments=pstring)
    exitCall = Call(name="exit", arguments=1)
    arrCond = Conditional(CondEq(filesArray, Macro('NULL')), [printfCall, exitCall])

    #Call open_thread_files
    open_thread_call = Call(name='open_thread_files', arguments=[filesArray, nthreads])

    # Build open section
    sec = Section("open", [arrCond, open_thread_call])

    # Close files array
    iSymbol = Symbol(name="i", dtype=np.int32)
    closeCall = Call(name="close", arguments=filesArray[iSymbol])
    iDim = CustomDimension(name="i", symbolic_size=nthreads)
    closeFilesIteration = Iteration(closeCall, iDim, nthreads - 1)
    closeSec = Section("close", closeFilesIteration)

    # Build write_size var
    write_size = Symbol(name="write_size", dtype=np.int64)
    time_M = Symbol(name="time_M", dtype=np.int32)
    time_m = Symbol(name="time_m", dtype=np.int32)
    u_vecSize1 = FieldFromPointer("size[1]", "u_vec")
    u_size = Symbol(name="u_size", dtype=np.int32)
    writeEq = IREq(write_size, ((time_M - time_m+1) * u_vecSize1 * u_size))
    cWriteEq = ClusterizedEq(writeEq, ispace=None)
    writeExp = Expression(cWriteEq, None, True)
    
    timerProfiler = Timer(profiler.name, [], ignoreDefinition=True)
    saveCall = Call(name='save', arguments=[nthreads, timerProfiler, write_size]) #save(nthreads, timers, write_size);

    symbs = FindSymbols("symbolics").visit(iet_body)
    dims = FindSymbols("dimensions").visit(iet_body)
    basics = FindSymbols("basics").visit(iet_body)
    bases = FindSymbols("indexedbases").visit(iet_body)
    defs = FindSymbols("defines").visit(iet_body)
    globals = FindSymbols("globals").visit(iet_body)

    uStencil = next((symb for symb in symbs if symb.name == "u"), None)
    t0 = next((dim for dim in dims if dim.name == "t0"), None)
    
    floatSize = Symbol(name="float_size", dtype=np.uint64)
    floatString = String(r"float")
    floatSizeInit = Call(name="sizeof", arguments=[floatString], retobj=floatSize)
    sizes = uStencil.symbolic_shape[2:]

    u_size = Symbol(name="u_size", dtype=np.uint64)
    UEq = IREq(u_size, (reduce(lambda x, y: x * y, sizes) * floatSize))
    cUEq = ClusterizedEq(UEq, ispace=None)
    UExp = Expression(cUEq, None, True)

    writeSection = write_build(nthreads, filesArray, iSymbol, u_size, uStencil, t0, uStencil.symbolic_shape[1])
    
    saveCallable = save_build(nthreads, timerProfiler, write_size)
    
    openThreadsCallable = open_threads_build(nthreads, filesArray, iSymbol, iDim)

    #import pdb; pdb.set_trace()

    iet_body.insert(0, UExp)
    iet_body.insert(0, floatSizeInit)
    iet_body.insert(0, sec)
    iet_body.append(closeSec)
    iet_body.append(writeSection)
    iet_body.append(writeExp)
    iet_body.append(saveCall)

    return iet_body

def write_build(nthreads, filesArray, iSymbol, u_size, uStencil, t0, uVecSize1):
    iDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    interval = Interval(iDim, 0, uVecSize1)
    intervalGroup = IntervalGroup((interval))
    ispace = IterationSpace(intervalGroup)
    itNodes = []

    tid = Symbol(name="tid", dtype=np.int32)
    tidEq = IREq(tid, Mod(iSymbol, nthreads))
    cTidEq = ClusterizedEq(tidEq, ispace=ispace)
    itNodes.append(Expression(cTidEq, None, True))
    
    ret = Symbol(name="ret", dtype=np.int32)
    writeCall = Call(name="write", arguments=[filesArray[tid], uStencil[t0, iSymbol], u_size], retobj=ret)
    itNodes.append(writeCall)

    pstring = String("'Cannot open output file'")
    condNodes = [Call(name="perror", arguments=pstring)]
    condNodes.append(Call(name="exit", arguments=1))
    cond = Conditional(CondNe(ret, u_size), condNodes)
    itNodes.append(cond)

    iDim = CustomDimension(name="i", symbolic_size=uVecSize1)
    pragma = cgen.Pragma("omp parallel for schedule(static,1) num_threads(nthreads)")
    writeIteration = Iteration(itNodes, iDim, uVecSize1-1, pragmas=[pragma])

    return Section("write", writeIteration)

    

def open_threads_build(nthreads, filesArray, iSymbol, iDim):
    nvme_id = Symbol(name="nvme_id", dtype=np.int32)
    ndisks = Symbol(name="NDISKS", dtype=np.int32)
    nvmeIdEq = IREq(nvme_id, Mod(iSymbol, ndisks))
    cNvmeIdEq = ClusterizedEq(nvmeIdEq, ispace=None) # ispace

    itNodes=[]
    itNodes.append(Expression(cNvmeIdEq, None, True))

    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)

    pstring = String(r"'data/nvme%d/thread_%d.data'")
    itNodes.append(Call(name="sprintf", arguments=[nameArray, pstring, nvme_id, iSymbol]))

    pstring = String(r"'Creating file %s\n'")
    itNodes.append(Call(name="printf", arguments=[pstring, nameArray]))

    ifNodes=[]
    pstring = String(r"'Cannot open output file\n'")
    ifNodes.append(Call(name="perror", arguments=pstring))

    ifNodes.append(Call(name="exit", arguments=1))

    opFlagsStr = String("OPEN_FLAGS")
    flagsStr = String("S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH")
    openCall = Call(name="open", arguments=[nameArray, opFlagsStr, flagsStr], retobj=filesArray[iSymbol])
    itNodes.append(openCall)

    openCond = Conditional(CondEq(filesArray[iSymbol], -1), ifNodes)
    
    itNodes.append(openCond)

    openIteration = Iteration(itNodes, iDim, nthreads-1)
    
    body = CallableBody(openIteration)
    callable = Callable("open_thread_files", body, "void", [filesArray, nthreads])

    return callable


def save_build(nthreads, timerProfiler, write_size):
    printfNodes = []
    pstring = String("'>>>>>>>>>>>>>> FORWARD <<<<<<<<<<<<<<<<<\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring]))

    pstring = String(r"'Threads %d\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, nthreads]))

    pstring = String(r"'Disks %d\n'")
    ndisksStr = String("NDISKS")
    printfNodes.append(Call(name="printf", arguments=[pstring, ndisksStr]))

    # Must retrieve section names from somewhere
    tSec0 = FieldFromPointer("section0", timerProfiler)
    tSec1 = FieldFromPointer("section1", timerProfiler)
    tSec2 = FieldFromPointer("section2", timerProfiler)
    tOpen = FieldFromPointer("open", timerProfiler)
    tWrite = FieldFromPointer("write", timerProfiler)
    tClose = FieldFromPointer("close", timerProfiler)

    pstring = String(r"'[FWD] Section0 %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tSec0]))

    pstring = String(r"'[FWD] Section1 %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tSec1]))

    pstring = String(r"'[FWD] Section2 %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tSec2])) 

    pstring = String(r"'[IO] Open %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tOpen]))

    pstring = String(r"'[IO] Write %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tWrite]))

    pstring = String(r"'[IO] Close %.2lf s\n'")
    printfNodes.append(Call(name="printf", arguments=[pstring, tClose]))
    
    nameDim = [CustomDimension(name="nameDim", symbolic_size=100)]
    nameArray = Array(name='name', dimensions=nameDim, dtype=np.byte)

    fileOpenNodes = []
    pstring = String(r"'fwd_disks_%d_threads_%d.csv'")
    fileOpenNodes.append(Call(name="sprintf", arguments=[nameArray, pstring, ndisksStr, nthreads]))

    pstring = String(r"'w'")
    filePointer = Pointer(name="ftp", dtype=FILE)
    fileOpenNodes.append(Call(name="fopen", arguments=[nameArray, pstring], retobj=filePointer))

    filePrintNodes = []
    pstring = String(r"'Disks, Threads, Bytes, [FWD] Section0, [FWD] Section1, [FWD] Section2, [IO] Open, [IO] Write, [IO] Close\n'")
    filePrintNodes.append(Call(name="fprintf", arguments=[filePointer, pstring]))
    
    pstring = String(r"'%d, %d, %ld, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n'")
    filePrintNodes.append(Call(name="fprintf", arguments=[filePointer, pstring, ndisksStr, nthreads, write_size,
                                                          tSec0, tSec1, tSec2, tOpen, tWrite, tClose]))

    saveCallBody = CallableBody(printfNodes+fileOpenNodes+filePrintNodes)
    saveCallable = Callable("save", saveCallBody, "void", [nthreads, timerProfiler, write_size])

    return saveCallable