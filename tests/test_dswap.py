import numpy as np
import pytest
import devito

from mpi4py import MPI
from devito import (Grid, TimeFunction, Function, Eq, Operator, Inc, CompressionConfig,
                    DiskSwapConfig, create_ds_path, remove_ds_path)
from devito import configuration, compiler_registry, TensorTimeFunction
from devito.arch.compiler import GNUCompiler



ALPHA = 1.01
C = 0.1
STEPS = 50


class TesteRegularMean(object):
    def test_regular_mean(self):
        grid = Grid(shape=(200, 200, 200))
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
        ds_path = create_ds_path("test_dswap_folder")
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
        
        remove_ds_path(ds_path)
        
        #RAM write and read
        write_op = Operator([eq_u, eq_v], opt=('advanced', {'disk-swap': None}),
                            name="write_op", language='openmp')
        write_op.apply()
        
        read_op = Operator(eq_w, opt=('advanced', {'disk-swap': None}),
                        name="read_op", language='openmp')
        read_op.apply()

        assert np.mean(w_func.data[-1]) == np.mean(w_ds_func.data[-1])

class TestMPIMean(object):

    @pytest.mark.parallel(mode=4)
    def test_mpi_means(self, mode):
        grid = Grid(shape=(200, 200, 200))
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        ds_path = create_ds_path("test_dswap_folder") if rank==0 else None
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


