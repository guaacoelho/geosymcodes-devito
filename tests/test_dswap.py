import numpy as np
import pytest

from mpi4py import MPI
from devito import (Grid, Function, TimeFunction, Eq, Operator, Inc, CompressionConfig,
                    DiskSwapConfig, create_ds_path, remove_ds_path)


ALPHA = 1.01
C = 0.1
STEPS = 50


@pytest.fixture(scope='module')
def ram_mean():
    grid = Grid(shape=(200, 200, 200))

    #RAM Functions, forward and backward terms
    u = TimeFunction(name="u", grid=grid, space_order=2, save=STEPS, time_order=2)
    u_next = u.forward
    ub = u.backward

    v = TimeFunction(name="v", grid=grid, space_order=2, save=STEPS, time_order=2)
    v_next = v.forward
    vb = v.backward

    p = TimeFunction(name="p", grid=grid, space_order=2, time_order=2)
    p_next = p.backward
    pf = p.forward

    w = Function(name='w', grid=grid, space_order=2)

    # Equations
    u_rhs = ALPHA*(u - ub) + C
    eq_u = Eq(u_next, u_rhs)

    v_rhs = ALPHA*(v - vb) + C
    eq_v = Eq(v_next, v_rhs)

    p_rhs = ALPHA*(p - pf) + C
    eq_p = Eq(p_next, p_rhs)

    eq_w = Inc(w, (u + v + p_next) * 1/ALPHA)

    #RAM write and read
    write_op = Operator([eq_u, eq_v], opt=('advanced', {'disk-swap': None}),
                        name="write_op", language='openmp')
    write_op.apply()

    read_op = Operator([eq_p, eq_w], opt=('advanced', {'disk-swap': None}),
                    name="read_op", language='openmp')
    read_op.apply()

    rmean = np.mean(w.data)
    assert rmean != 0
    assert (not np.isnan(rmean))
        
    yield rmean


'''
def test_regular_mean(ram_mean):
    grid = Grid(shape=(200, 200, 200))

    #Functions, forward and backward terms
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=2)
    u_next = u.forward
    ub = u.backward

    v = TimeFunction(name="v", grid=grid, space_order=2, time_order=2)
    v_next = v.forward
    vb = v.backward

    p = TimeFunction(name="p", grid=grid, space_order=2, time_order=2)
    p_next = p.backward
    pf = p.forward

    w = Function(name='w', grid=grid, space_order=2)

    # Equations
    u_rhs = ALPHA*(u - ub) + C
    eq_u = Eq(u_next, u_rhs)

    v_rhs = ALPHA*(v - vb) + C
    eq_v = Eq(v_next, v_rhs)

    p_rhs = ALPHA*(p - pf) + C
    eq_p = Eq(p_next, p_rhs)

    eq_w = Inc(w, (u + v + p_next) * 1/ALPHA)

    #DISK write and read
    ds_path = create_ds_path("test_dswap_folder")
    write_config = DiskSwapConfig(functions=[u, v],
                                mode="write",
                                path=ds_path,
                                odirect=1)

    write_op = Operator([eq_u, eq_v], opt=('advanced', {'disk-swap': write_config}),
                        name="write_op", language='openmp')
    write_op.apply(time_m=1, time_M=STEPS-2)

    read_config = DiskSwapConfig(functions=[u, v],
                                mode="read",
                                path=ds_path,
                                odirect=1)

    read_op = Operator([eq_p, eq_w], opt=('advanced', {'disk-swap': read_config}),
                        name="read_op", language='openmp')
    read_op.apply(time_m=1, time_M=STEPS-2)

    remove_ds_path(ds_path)

    ds_mean = np.mean(w.data)
    assert ds_mean != 0
    assert (not np.isnan(ds_mean))

    assert ram_mean == ds_mean
'''




@pytest.mark.parallel(mode=4)
def test_mpi_means(mode):
    grid = Grid(shape=(200, 200, 200))
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ds_path = create_ds_path("dswap_temp") if rank==0 else None
    ds_path = comm.bcast(ds_path, root=0)
    
    # DSWAP functions, forward and backward terms
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=2)
    u_next = u.forward
    ub = u.backward

    v = TimeFunction(name="v", grid=grid, space_order=2, time_order=2)
    v_next = v.forward
    vb = v.backward

    p = TimeFunction(name="p", grid=grid, space_order=2, time_order=2)
    p_next = p.backward
    pf = p.forward

    w = Function(name='w', grid=grid, space_order=2)

    # DSWAP Equations
    u_rhs = ALPHA*(u - ub) + C
    eq_u = Eq(u_next, u_rhs)

    v_rhs = ALPHA*(v - vb) + C
    eq_v = Eq(v_next, v_rhs)

    p_rhs = ALPHA*(p - pf) + C
    eq_p = Eq(p_next, p_rhs)

    eq_w = Inc(w, (u + v + p_next) * 1/ALPHA)

    # DSWAP write and read
    write_config = DiskSwapConfig(functions=[u, v],
                                mode="write",
                                path=ds_path,
                                odirect=1)

    write_op = Operator([eq_u, eq_v], opt=('advanced', {'disk-swap': write_config}),
                        name="write_op", language='openmp')
    write_op.apply(time_m=1, time_M=STEPS-2)

    read_config = DiskSwapConfig(functions=[u, v],
                                mode="read",
                                path=ds_path,
                                odirect=1)

    read_op = Operator([eq_p, eq_w], opt=('advanced', {'disk-swap': read_config}),
                        name="read_op", language='openmp')
    read_op.apply(time_m=1, time_M=STEPS-2)
    
    comm.Barrier()
    if rank == 0:
        remove_ds_path(ds_path)

    '''
    #RAM Functions, forward and backward terms
    u_ram = TimeFunction(name="u_ram", grid=grid, space_order=2, save=STEPS, time_order=2)
    ur_next = u_ram.forward
    urb = u_ram.backward

    v_ram = TimeFunction(name="v_ram", grid=grid, space_order=2, save=STEPS, time_order=2)
    vr_next = v_ram.forward
    vrb = v_ram.backward

    p_ram = TimeFunction(name="p_ram", grid=grid, space_order=2, time_order=2)
    pr_next = p_ram.backward
    prf = p_ram.forward

    w_ram = Function(name='w_ram', grid=grid, space_order=2)

    # Equations
    ur_rhs = ALPHA*(u_ram - urb) + C
    eq_ur = Eq(ur_next, ur_rhs)

    vr_rhs = ALPHA*(v_ram - vrb) + C
    eq_vr = Eq(vr_next, vr_rhs)

    pr_rhs = ALPHA*(p_ram - prf) + C
    eq_pr = Eq(pr_next, pr_rhs)

    eq_wr = Inc(w_ram, (u_ram + v_ram + pr_next) * 1/ALPHA)

    #RAM write and read
    write_opr = Operator([eq_ur, eq_vr], opt=('advanced', {'disk-swap': None}),
                        name="write_opr", language='openmp')
    write_opr.apply()

    read_opr = Operator([eq_pr, eq_wr], opt=('advanced', {'disk-swap': None}),
                    name="read_opr", language='openmp')
    read_opr.apply()
    '''
    print("ram mean:", 0, " dsmean:", np.mean(w.data))
    #assert np.mean(w_func.data[-1]) == np.mean(w_ds_func.data[-1])


'''
@pytest.mark.parallel(mode=4)
def test_mpi_means(mode):
    grid = Grid(shape=(200, 200, 200))
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ds_path = create_ds_path("dswap_temp") if rank==0 else None
    ds_path = comm.bcast(ds_path, root=0)
        
    t = grid.stepping_dim
    x, y, z = grid.dimensions
    
    #RAM Functions and equations
    u_func = TimeFunction(name="u", grid=grid, space_order=2, save=STEPS+1)
    v_func = TimeFunction(name="v", grid=grid, space_order=2, save=STEPS+1)
    w_func = TimeFunction(name='w', grid=grid, space_order=2)
    
    eq_u = Eq(u_func[t+1, x, y, z], ALPHA*u_func[t, x, y, z] + C)
    eq_v = Eq(v_func[t+1, x, y, z], ALPHA*v_func[t, x, y, z] + C)
    eq_w = Inc(w_func, (v_func + u_func) * 1/ALPHA)
    
    #DISK Functions and equations
    u_ds_func = TimeFunction(name="u", grid=grid, space_order=2)
    v_ds_func = TimeFunction(name="v", grid=grid, space_order=2)
    w_ds_func = TimeFunction(name='w', grid=grid, space_order=2)

    eq_ds_u = Eq(u_ds_func[t+1, x, y, z], ALPHA*u_ds_func[t, x, y, z] + C)
    eq_ds_v = Eq(v_ds_func[t+1, x, y, z], ALPHA*v_ds_func[t, x, y, z] + C)
    eq_ds_w = Inc(w_ds_func, (v_ds_func + u_ds_func) * 1/ALPHA)

    #DISK write and read
    write_config = DiskSwapConfig(functions=[u_ds_func, v_ds_func],
                                mode="write",
                                compression=False,
                                path=ds_path,
                                odirect=1)

    write_ds_op = Operator([eq_ds_u, eq_ds_v], opt=('advanced', {'disk-swap': write_config}),
                        name="write_ds_op", language='openmp')
    write_ds_op.apply(time_M=STEPS)
    
    read_config = DiskSwapConfig(functions=[u_ds_func, v_ds_func],
                                mode="read",
                                compression=False,
                                path=ds_path,
                                odirect=1)

    read_ds_op = Operator(eq_ds_w, opt=('advanced', {'disk-swap': read_config}),
                        name="read_ds_op", language='openmp')
    read_ds_op.apply(time_M=STEPS)
    
    comm.Barrier()
    if rank == 0:
        remove_ds_path(ds_path)
    
    #RAM write and read
    write_op = Operator([eq_u, eq_v], opt=('advanced', {'disk-swap': None}),
                        name="write_op", language='openmp')
    write_op.apply()
    
    read_op = Operator(eq_w, opt=('advanced', {'disk-swap': None}),
                    name="read_op", language='openmp')
    read_op.apply()
        
    assert np.mean(w_func.data[-1]) == np.mean(w_ds_func.data[-1])
'''

'''
@pytest.mark.parametrize('mode, value',
                         [("lossless", None),
                          ('rate', 1),
                          ('rate', 2),
                          ('rate', 10),
                          ('precision', 2),
                          ('precision', 10),
                          ('accuracy', 0.1),
                          ('accuracy', 1),
                          ('accuracy', 2),
                          ('accuracy', 10),])
def test_compression_means(ram_mean, mode, value):
    grid = Grid(shape=(200, 200, 200))
    t = grid.stepping_dim
    x, y, z = grid.dimensions
    
    #DISK Functions and equations
    u_ds_func = TimeFunction(name="u", grid=grid, space_order=2)
    v_ds_func = TimeFunction(name="v", grid=grid, space_order=2)
    w_ds_func = TimeFunction(name='w', grid=grid, space_order=2)

    eq_ds_u = Eq(u_ds_func[t+1, x, y, z], ALPHA*u_ds_func[t, x, y, z] + C)
    eq_ds_v = Eq(v_ds_func[t+1, x, y, z], ALPHA*v_ds_func[t, x, y, z] + C)
    eq_ds_w = Inc(w_ds_func, (v_ds_func + u_ds_func) * 1/ALPHA)

    #Compression configuration
    value_type = "RATE" if mode == "rate" else "value"
    compression_args = {"method":mode, value_type:value}
    cc = CompressionConfig(**compression_args)
    
    #DISK write and read
    ds_path = create_ds_path("dswap_folder")
    write_config = DiskSwapConfig(functions=[u_ds_func, v_ds_func],
                                mode="write",
                                compression=cc,
                                path=ds_path,
                                odirect=1)

    write_ds_op = Operator([eq_ds_u, eq_ds_v], opt=('advanced', {'disk-swap': write_config}),
                        name="write_ds_op", language='openmp')
    write_ds_op.apply(time_M=STEPS)
    
    read_config = DiskSwapConfig(functions=[u_ds_func, v_ds_func],
                                mode="read",
                                compression=cc,
                                path=ds_path,
                                odirect=1)

    read_ds_op = Operator(eq_ds_w, opt=('advanced', {'disk-swap': read_config}),
                        name="read_ds_op", language='openmp')
    read_ds_op.apply(time_M=STEPS)
    
    remove_ds_path(ds_path)

    ds_mean = np.mean(w_ds_func.data[-1])

    assert (not np.isnan(ds_mean))
'''