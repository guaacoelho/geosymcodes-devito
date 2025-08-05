from devito import (Eq, Operator, VectorTimeFunction, TensorTimeFunction,
                    Function, TimeFunction)
from devito import solve
from examples.seismic import PointSource, Receiver
from examples.seismic.stiffness.utils import D, S, vec, C_Matrix, gather
from examples.seismic.utils import get_ooc_config


def src_rec(v, tau, model, geometry, forward=True):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)
    name = "rec_tau" if forward else "rec"
    rec = Receiver(name="%s" % name, grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)
    tau = vec(tau)
    if forward:

        # The source injection term
        src_xx = src.inject(field=tau[0].forward, expr=src * s)
        src_zz = src.inject(field=tau[1].forward, expr=src * s)
        src_expr = src_xx + src_zz
        if model.grid.dim == 3:
            src_yy = src.inject(field=tau[2].forward, expr=src * s)
            src_expr += src_yy
        # Create interpolation expression for receivers
        rec_term_vx = rec_vx.interpolate(expr=v[0])
        rec_term_vz = rec_vz.interpolate(expr=v[-1])
        expr = tau[0] + tau[1]
        rec_expr = rec_term_vx + rec_term_vz
        if model.grid.dim == 3:
            expr += tau[2]
            rec_term_vy = rec_vy.interpolate(expr=v[1])
            rec_expr += rec_term_vy
        rec_term_tau = rec.interpolate(expr=expr)
        rec_expr += rec_term_tau

    else:
        # Construct expression to inject receiver values
        rec_xx = rec.inject(field=tau[0].backward, expr=rec*s)
        rec_zz = rec.inject(field=tau[1].backward, expr=rec*s)
        rec_expr = rec_xx + rec_zz
        expr = tau[0] + tau[1]
        if model.grid.dim == 3:
            rec_expr += rec.inject(field=tau[2].backward, expr=rec*s)
            expr += tau[2]
        # Create interpolation expression for the adjoint-source
        src_expr = src.interpolate(expr=expr)

    return src_expr, rec_expr


def elastic_stencil(model, v, tau, forward=True, par='lam-mu'):

    damp = model.damp

    rho = model.rho

    C = C_Matrix(model, par)

    tau = vec(tau)
    if forward:

        pde_v = rho * v.dt - D(tau)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        pde_tau = tau.dt - C * S(v.forward)
        u_t = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

        return [u_v, u_t]

    else:

        """
        Implementation of the elastic wave-equation from:
        1 - Feng and Schuster (2017): Elastic least-squares reverse time migration
        https://doi.org/10.1190/geo2016-0254.1
        """

        pde_v = rho * v.dtl - D(C.T*tau)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        pde_tau = -tau.dtl + S(v.backward)
        u_t = Eq(tau.backward, damp * solve(pde_tau, tau.backward))

        return [u_v, u_t]


def EqsLamMu(model, sig, u, v, grad_lam, grad_mu, grad_rho, C, space_order=8, **kwargs):
    hl = TimeFunction(name='hl', grid=model.grid, space_order=space_order,
                      time_order=1)
    hm = TimeFunction(name='hm', grid=model.grid, space_order=space_order,
                      time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wl = gather(0, C.dlam * S(v))
    Wm = gather(0, C.dmu * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wl_update = Eq(hl, Wl.T * W2)
    gradient_lam = Eq(grad_lam, grad_lam + hl)

    wm_update = Eq(hm, Wm.T * W2)
    gradient_mu = Eq(grad_mu, grad_mu + hm)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wl_update, gradient_lam, wm_update, gradient_mu, wr_update, gradient_rho]


def EqsPhiCC(model, sig, u, v, grad_phi, grad_CC, grad_rho, C, space_order=8, **kwargs):
    hphi = TimeFunction(name='hphi', grid=model.grid, space_order=space_order,
                      time_order=1)
    hCC = TimeFunction(name='hCC', grid=model.grid, space_order=space_order,
                      time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wphi = gather(0, -C.dphi * S(v))
    WCC = gather(0, -C.dCC * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wphi_update = Eq(hphi, Wphi.T * W2)
    gradient_phi = Eq(grad_phi, grad_phi - hphi)

    wCC_update = Eq(hCC, WCC.T * W2)
    gradient_CC = Eq(grad_CC, grad_CC - hCC)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wphi_update, gradient_phi, wCC_update, gradient_CC, wr_update, gradient_rho]


def EqsVpVsRho(model, sig, u, v, grad_vp, grad_vs, grad_rho, C, space_order=8, **kwargs):
    hvp = TimeFunction(name='hvp', grid=model.grid, space_order=space_order,
                       time_order=1)
    hvs = TimeFunction(name='hvs', grid=model.grid, space_order=space_order,
                       time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wvp = gather(0, -C.dvp * S(v))
    Wvs = gather(0, -C.dvs * S(v))
    Wr = gather(v.dt, - C.drho * S(v))

    W2 = gather(u, sig)

    wvp_update = Eq(hvp, Wvp.T * W2)
    gradient_lam = Eq(grad_vp, grad_vp - hvp)

    wvs_update = Eq(hvs, Wvs.T * W2)
    gradient_mu = Eq(grad_vs, grad_vs - hvs)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wvp_update, gradient_lam, wvs_update, gradient_mu, wr_update, gradient_rho]


def EqsIpIs(model, sig, u, v, grad_Ip, grad_Is, grad_rho, C, space_order=8, **kwargs):

    hIp = TimeFunction(name='hIp', grid=model.grid, space_order=space_order,
                       time_order=1)

    hIs = TimeFunction(name='hIs', grid=model.grid, space_order=space_order,
                       time_order=1)

    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    WIp = gather(0, C.dIp * S(v))
    WIs = gather(0, C.dIs * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wIp_update = Eq(hIp, WIp.T * W2)
    gradient_Ip = Eq(grad_Ip, grad_Ip + hIp)

    wIs_update = Eq(hIs, WIs.T * W2)
    gradient_Is = Eq(grad_Is, grad_Is + hIs)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wIp_update, gradient_Ip, wIs_update, gradient_Is, wr_update, gradient_rho]


def EqsC11C12C33(model, sig, u, v, grad_C11, grad_C12, grad_C33, C, space_order=8, **kwargs):
    hC11 = TimeFunction(name='hC11', grid=model.grid, space_order=space_order,
                        time_order=1)
    hC12 = TimeFunction(name='hC12', grid=model.grid, space_order=space_order,
                        time_order=1)
    hC33 = TimeFunction(name='hC33', grid=model.grid, space_order=space_order,
                        time_order=1)
 
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)
 
    grad_rho = kwargs["grad4"]

    WC11 = gather(0, -C.dC11 * S(v))
    WC12 = gather(0, -C.dC12 * S(v))
    WC33 = gather(0, - C.dC33 * S(v))
    Wr = gather(v.dt, 0)
 
    W2 = gather(u, sig)
 
    wC11_update = Eq(hC11, WC11.T * W2)
    gradient_C11 = Eq(grad_C11, grad_C11 - hC11)
 
    wC12_update = Eq(hC12, WC12.T * W2)
    gradient_C12 = Eq(grad_C12, grad_C12 - hC12)
 
    wC33_update = Eq(hC33, WC33.T * W2)
    gradient_C33 = Eq(grad_C33, grad_C33 - hC33)
 
    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)
 
    return [wC11_update, gradient_C11, wC12_update,
            gradient_C12, wC33_update, gradient_C33, wr_update, gradient_rho]


def ForwardOperator(model, geometry, space_order=4, save=False, par='lam-mu', **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """

    dswap = kwargs.get("dswap", False)

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save and not dswap else None,
                           space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             space_order=space_order, time_order=1)

    if dswap:
        kwargs.update(get_ooc_config(v, "write", **kwargs))

    eqn = elastic_stencil(model, v, tau, par=par)

    src_expr, rec_expr = src_rec(v, tau, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name="ForwardGenericElastic", **kwargs)


def AdjointOperator(model, geometry, space_order=4, par='lam-mu', **kwargs):
    """
    Construct an adjoint modelling operator in a viscoacoustic medium.
    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """

    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)

    src_expr, rec_expr = src_rec(u, sig, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name='AdjointGenericElastic', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True, par='lam-mu', **kwargs):
    """
    Construct a gradient operator in an elastic media.
    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    """

    dswap = kwargs.get("dswap", False)

    # Gradient symbol and wavefield symbols
    grad1 = Function(name='grad1', grid=model.grid)
    grad2 = Function(name='grad2', grid=model.grid)
    grad3 = Function(name='grad3', grid=model.grid)
    grad4 = Function(name='grad4', grid=model.grid)

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save and not dswap else None,
                           space_order=space_order, time_order=1)
    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)

    s = model.grid.time_dim.spacing
    rho = model.rho

    if dswap:
        kwargs.update(get_ooc_config(v, "read", **kwargs))

    C = C_Matrix(model, par)

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)
    sig = vec(sig)

    kernel = kernels[par]
    gradient_update = kernel(model, sig, u, v, grad1, grad2,
                             grad3, C, space_order=space_order, grad4=grad4)

    # Construct expression to inject receiver values
    rec_term_vx = rec_vx.inject(field=u[0].backward, expr=s*rec_vx/rho)
    rec_term_vz = rec_vz.inject(field=u[-1].backward, expr=s*rec_vz/rho)
    rec_expr = rec_term_vx + rec_term_vz
    if model.grid.dim == 3:
        rec_expr += rec_vy.inject(field=u[1].backward, expr=s*rec_vy/rho)

    if kwargs.pop('has_rec_p'):
        rec_p = Receiver(name='rec_p', grid=model.grid, time_range=geometry.time_axis,
                         npoint=geometry.nrec)
        rec_term_sigx = rec_p.inject(field=sig[0].backward, expr=s*rec_p/rho)
        rec_term_sigz = rec_p.inject(field=sig[1].backward, expr=s*rec_p/rho)
        rec_expr += rec_term_sigx + rec_term_sigz
        if model.grid.dim == 3:
            rec_expr += rec_p.inject(field=sig[2].backward, expr=rec_p/rho)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + rec_expr + gradient_update, subs=model.spacing_map,
                    name='GradientElastic', **kwargs)


kernels = {'lam-mu': EqsLamMu, 'vp-vs-rho': EqsVpVsRho, 'Ip-Is-rho': EqsIpIs,
           'Iso-C11C12C33': EqsC11C12C33, 'rho-phi-CC':EqsPhiCC}
