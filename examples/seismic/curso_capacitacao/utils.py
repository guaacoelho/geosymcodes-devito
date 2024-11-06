import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from devito import (Eq, Operator, VectorTimeFunction, TimeFunction,Function, NODE, div, grad)
from examples.seismic import RickerSource, AcquisitionGeometry
from examples.seismic.viscoacoustic.operators import *


def image_show(model, data1, data2, data3, data4, vmin1, vmax1):
    

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 13))
    
    slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]

    img1 = ax[0][0].imshow(np.transpose(np.diff(data1.data[slices], axis=1)), vmin=vmin1, vmax=vmax1,
                        cmap='gray')
    ax1=plt.gca()
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1)
    
    ax[0][0].set_title("Acústico", fontsize=20)
    ax[0][0].set_xlabel('X (m)', fontsize=20)
    ax[0][0].set_ylabel('Depth (m)', fontsize=20)
    ax[0][0].set_aspect('auto')

    img2 = ax[0][1].imshow(np.transpose(np.diff(data2.data[slices], axis=1)), vmin=vmin1, vmax=vmax1,
                        cmap='gray')
    fig.colorbar(img2, cax=cax1)
    ax[0][1].set_title("SLS", fontsize=20)
    ax[0][1].set_xlabel('X (m)', fontsize=20)
    ax[0][1].set_ylabel('Depth (m)', fontsize=20)
    ax[0][1].set_aspect('auto')

    img3 = ax[1][0].imshow(np.transpose(np.diff(data3.data[slices], axis=1)), vmin=vmin1, vmax=vmax1,
                        cmap='gray')
    fig.colorbar(img3, cax=cax1)
    ax[1][0].set_title("Kelvin-Voigt", fontsize=20)
    ax[1][0].set_xlabel('X (m)', fontsize=20)
    ax[1][0].set_ylabel('Depth (m)', fontsize=20)
    ax[1][0].set_aspect('auto')

    img4 = ax[1][1].imshow(np.transpose(np.diff(data2.data[slices], axis=1)), vmin=vmin1, vmax=vmax1,
                        cmap='gray')
    fig.colorbar(img4, cax=cax1)
    ax[1][1].set_title("Max", fontsize=20)
    ax[1][1].set_xlabel('X (m)', fontsize=20)
    ax[1][1].set_ylabel('Depth (m)', fontsize=20)
    ax[1][1].set_aspect('auto')    
    
    plt.tight_layout()
    plt.show()
    
    

def plot(model, a, amax, time_range, nsnaps, title=None):
    
    nbl = model.nbl
    origin = model.origin
    spacing = model.spacing
    nxpad, nzpad = model.shape[0] + 2 * nbl, model.shape[1] + 2 * nbl
    shape_pad = np.array(model.shape) + 2 * nbl
    origin_pad = tuple([o - s*nbl for o, s in zip(origin, spacing)])
    extent_pad = tuple([s*(n-1) for s, n in zip(spacing, shape_pad)])
    # Note: flip sense of second dimension to make the plot positive downwards
    plt_extent = [origin_pad[0], origin_pad[0] + extent_pad[0],
                  origin_pad[1] + extent_pad[1], origin_pad[1]]
    
    # Plot the wavefields, each normalized to scaled maximum of last time step

    
    dt=time_range.step
    factor = round((time_range.num-1) / nsnaps)

    fig, axes = plt.subplots(1, 4, figsize=(25, 4), sharex=True)
    fig.suptitle(title, size=20)
    for count, ax in enumerate(axes.ravel()):
        snapshot = factor * (count + 1)
        ax.imshow(np.transpose(a.data[snapshot, :, :]), cmap="seismic", vmin=-amax,
                  vmax=+amax, extent=plt_extent)
        ax.plot(model.domain_size[0] * .5, 10, 'red', linestyle='None', marker='*',
                markersize=8, label="Source")
        ax.grid()
        ax.tick_params('both', length=4, width=0.5, which='major', labelsize=10)
        ax.set_title("Campo de onda t=%.2fms" % (factor*count*dt), fontsize=10)
        ax.set_xlabel("X Coordinate (m)", fontsize=10)
        ax.set_ylabel("Z Coordinate (m)", fontsize=10)


def V_Q_plot(model):
    aspect_ratio = model.shape[0]/model.shape[1]

    plt_options_model = {'cmap': 'jet', 'extent': [model.origin[0], model.origin[0] + model.domain_size[0],
                                                   model.origin[1] + model.domain_size[1], model.origin[1]]}
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]
    a = model.vp.data[slices]
    b = model.qp.data[slices]
    c = 1./model.b.data[slices]
    img1 = ax[0].imshow(np.transpose(a), vmin=a.min(), vmax=a.max(), **plt_options_model)
    fig.colorbar(img1, ax=ax[0])
    ax[0].set_title(r"V (km/s)", fontsize=20)
    ax[0].set_xlabel('X (m)', fontsize=20)
    ax[0].set_ylabel('Depth (m)', fontsize=20)
    ax[0].set_aspect('auto')

    img2 = ax[1].imshow(np.transpose(b), vmin=b.min(), vmax=b.max(), **plt_options_model)
    fig.colorbar(img2, ax=ax[1])
    ax[1].set_title(r"Q", fontsize=20)
    ax[1].set_xlabel('X (m)', fontsize=20)
    ax[1].set_ylabel('Depth (m)', fontsize=20)
    ax[1].set_aspect('auto')

    img3 = ax[2].imshow(np.transpose(c), vmin=c.min(), vmax=c.max(),**plt_options_model)
    fig.colorbar(img3, ax=ax[2])
    ax[2].set_title(r"$\rho (g/cm^3)$", fontsize=20)
    ax[2].set_xlabel('X (m)', fontsize=20)
    ax[2].set_ylabel('Depth (m)', fontsize=20)
    ax[2].set_aspect('auto')
    
    plt.tight_layout()
    plt.show()

def plot_shot(rec1, rec2, rec3, rec4, model, t0, tn, colorbar=True):
    
    scale = np.max(rec1) / 70.
    
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0], 1e-3*tn, t0]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))

    img1 = ax[0].imshow(rec1, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    fig.colorbar(img1, ax=ax[0])
    ax[0].set_title(r"Acústico", fontsize=10)
    ax[0].set_xlabel('Afastamento (km)', fontsize=10)
    ax[0].set_ylabel('Tempo (s)', fontsize=10)
    ax[0].set_aspect('auto')

    img2 = ax[1].imshow(rec2, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    fig.colorbar(img2, ax=ax[1])
    ax[1].set_title(r"SLS", fontsize=10)
    ax[1].set_xlabel('Afastamento (km)', fontsize=10)
    ax[1].set_ylabel('Tempo (s)', fontsize=10)
    ax[1].set_aspect('auto')

    img3 = ax[2].imshow(rec3, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    fig.colorbar(img3, ax=ax[2])
    ax[2].set_title("Maxwell", fontsize=10)
    ax[2].set_xlabel('Afastamento (km)', fontsize=10)
    ax[2].set_ylabel('Tempo (s)', fontsize=10)
    ax[2].set_aspect('auto')
    
    img3 = ax[3].imshow(rec4, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    fig.colorbar(img3, ax=ax[3])
    ax[3].set_title("Kelvin-Voigt", fontsize=10)
    ax[3].set_xlabel('Afastamento (km)', fontsize=10)
    ax[3].set_ylabel('Tempo (s)', fontsize=10)
    ax[3].set_aspect('auto')
    plt.show()
    plt.tight_layout()
    #     plt.show()


def plot_cmp_rec_one(time, trace_b, trace_r, trace_d, trace_a, ):
   
    plt.figure(figsize=(16, 10))
    plt.subplot(2,1,1)
    plt.plot(time, trace_b, '-b', label='Acústico')
    plt.plot(time, trace_r, '-g', label='SLS')
    plt.plot(time, trace_d, '-y', label='Maxwell')
    plt.plot(time, trace_a, '-m', label='Kelvin-Voigt')
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Amplitude')
    plt.legend()


def plot_shotrecord_utils(rec, model, t0, tn, scale, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def lapla (model, image):
    lapla = Function(name='lapla', grid=model.grid, space_order = 8)
    stencil=Eq(lapla, image.laplace)
    op = Operator([stencil])
    op.apply()
    
    return lapla


def acoustic_2nd_order(model, geometry, field2, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    space_order = kwargs.get('space_order')
    #     op_type = kwargs.get('op_type')
    forward = kwargs.get('forward', True)
    s = model.grid.stepping_dim.spacing
    vp = model.vp
    b = model.b
    damp = model.damp
        
    # Particle Velocity
    #     field1 = kwargs.pop('v')

    # Density
    rho = 1. / b

    bm = rho * (vp * vp)

    if forward:

        pde_p = 2. * field2 - damp * field2.backward + s * s * bm * div(b*grad(field2, shift=.5), shift=-.5)
        u_p = Eq(field2.forward, damp * pde_p)

        #return [u_aux2, u_p]
        return [u_p]

    else:

        pde_q = 2. * field2 - damp * field2.forward + s * s * div(b * grad(bm * field2, shift=.5), shift=-.5)
        u_q = Eq(field2.backward, damp * pde_q)

        return [u_q]


def src_rec1(p, model, time_range, f0, dt):
    
    src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range, t0=1/f0)
    expr= src*dt**2 / model.m # source expression for inject

    # Primeiro, posicionareoms a fonte centralmente e, em seguida, definiremos a profundidade.
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 20.  # Profundidade em 20m

    rec = Receiver(name='rec', grid=model.grid, npoint=model.shape[0], time_range=time_range)

    rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=model.shape[0])
    rec.coordinates.data[:, 1] = 20.  # Profundidade 20m

    src_term = src.inject(field=p.forward, expr=expr)
    rec_term = rec.interpolate(expr=p)

    return src_term + rec_term, src, rec, p
    
    
def modelling(model, time_range, f0, dt, **kwargs):

    time_order = kwargs.get('time_order')
    kernel = kwargs.get('kernel') 
    sl = kwargs.get('sl', [0, 0]) 

    space_order = model.space_order
    
    p = TimeFunction(name="p", grid=model.grid, time_order=time_order, 
                     space_order=space_order, staggered=NODE, save=time_range.num)
          
    src_rec_expr, src, rec, p = src_rec1(p, model, time_range, f0, dt)
    
    geometry = AcquisitionGeometry(model, rec.coordinates.data, src.coordinates.data, 
                                   0, int(time_range.stop), f0=f0, src_type='Ricker')
    
    if sl[1] != 0:
        geometry.src_positions[0, :] = sl
   
    
    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, p, space_order=space_order)
        
    op = Operator(eqn + src_rec_expr, subs=model.spacing_map)
    op(dt=dt, src=src, rec=rec)
    
    return rec, p

kernels = {'sls2': sls_2nd_order,'kv2': kv_2nd_order, 'max2': maxwell_2nd_order, 'acoustic2':acoustic_2nd_order}