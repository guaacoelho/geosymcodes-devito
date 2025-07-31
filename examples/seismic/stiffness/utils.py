from devito.types.tensor import (TensorFunction, TensorTimeFunction,
                                 VectorFunction, VectorTimeFunction, tens_func)
import numpy as np
import copy
from math import ceil
from sympy import symbols, Matrix, ones, zeros, simplify
from sympy.calculus.finite_diff import differentiate_finite
import numbers


class C_Matrix():

    C_matrix_dependency = {'lam-mu': 'C_lambda_mu', 'vp-vs-rho': 'C_vp_vs_rho',
                           'Ip-Is-rho': 'C_Ip_Is_rho', 'C-elements': 'C_from_model',
                           'Iso-C11C12C33':'CISO_from_model'}

    def __new__(cls, model, parameters):
        cls.model = model # TODO: verify another way to make model available to derivative calculation
        c_m_gen = cls.C_matrix_gen(parameters)
        return c_m_gen(model)

    @classmethod
    def C_matrix_gen(cls, parameters):
        return getattr(cls, cls.C_matrix_dependency[parameters])

    def _matrix_init(dim, asymmetrical=False, full_matrix=False):
        def cij(ii, jj):
            if full_matrix:
                # If you want the matrix to be fully populated with symbolic elements
                return symbols('C%s%s' % (jj, ii))
            if not asymmetrical:
                # It reorders the indices so that the smaller one comes first
                # (e.g., C₂₁ becomes C₁₂), thereby enforcing matrix symmetry
                jj, ii = min(ii, jj), max(ii, jj)
            if (ii == jj or (ii <= dim and jj <= dim)):
                return symbols('C%s%s' % (jj, ii))
            return 0

        d = dim*2 + dim-2
        Cij = [[cij(i, j) for i in range(1, d)] for j in range(1, d)]
        return AuxiliaryMatrix(Cij)

    @classmethod
    def C_from_model(cls, model):
        def subsC():
            # Generation of the complete symbolic matrix
            dict_C = {'C11': getattr(model, 'C11'),
                      'C22': getattr(model, 'C22'),
                      'C33': getattr(model, 'C33'),
                      'C32': getattr(model, 'C32'),
                      'C23': getattr(model, 'C23'),
                      'C12': getattr(model, 'C12'),
                      'C21': getattr(model, 'C21'),
                      'C13': getattr(model, 'C13'),
                      'C31': getattr(model, 'C31')}
            if model.dim == 3:
                dict_C['C14'] = getattr(model, 'C14')
                dict_C['C15'] = getattr(model, 'C15')
                dict_C['C16'] = getattr(model, 'C16')
                dict_C['C24'] = getattr(model, 'C24')
                dict_C['C25'] = getattr(model, 'C25')
                dict_C['C26'] = getattr(model, 'C26')
                dict_C['C34'] = getattr(model, 'C34')
                dict_C['C35'] = getattr(model, 'C35')
                dict_C['C36'] = getattr(model, 'C36')
                dict_C['C41'] = getattr(model, 'C41')
                dict_C['C42'] = getattr(model, 'C42')
                dict_C['C43'] = getattr(model, 'C43')
                dict_C['C44'] = getattr(model, 'C44')
                dict_C['C45'] = getattr(model, 'C45')
                dict_C['C46'] = getattr(model, 'C46')
                dict_C['C51'] = getattr(model, 'C51')
                dict_C['C52'] = getattr(model, 'C52')
                dict_C['C53'] = getattr(model, 'C53')
                dict_C['C54'] = getattr(model, 'C54')
                dict_C['C55'] = getattr(model, 'C55')
                dict_C['C56'] = getattr(model, 'C56')
                dict_C['C61'] = getattr(model, 'C61')
                dict_C['C62'] = getattr(model, 'C62')
                dict_C['C63'] = getattr(model, 'C63')
                dict_C['C64'] = getattr(model, 'C64')
                dict_C['C65'] = getattr(model, 'C65')
                dict_C['C66'] = getattr(model, 'C66')

            return dict_C

        matriz = C_Matrix._matrix_init(model.dim, asymmetrical=True, full_matrix=True)
        subs = subsC()
        M = matriz.subs(subs)
        return M

    @staticmethod
    def CISO_from_model(model):
        def subsC():
            # Generation of the symbolic matrix in the Iso-C11C12C33 format
            dict_C = {'C11': getattr(model, 'C11', 0),
                      'C22': getattr(model, 'C11', 0),
                      'C33': getattr(model, 'C33', 0),
                      'C12': getattr(model, 'C12', 0)}
            if model.dim == 3:
                dict_C['C44'] = getattr(model, 'C44', 0)
                dict_C['C55'] = getattr(model, 'C55', 0)
                dict_C['C66'] = getattr(model, 'C66', 0)
                dict_C['C13'] = getattr(model, 'C13', 0)
                dict_C['C23'] = getattr(model, 'C23', 0)
            return dict_C

        matrix = C_Matrix._matrix_init(model.dim)
        subs = subsC()
        M = matrix.subs(subs)
        return M

    @staticmethod
    def _generate_derivative(derivative, symbolic_matrix):
        # Gets the name of the element being used to calculate
        # the derivative (removing the 'd' from the beginning)
        element = derivative[1:]

        matrix = zeros(*symbolic_matrix.shape)
        symbol = getattr(C_Matrix.model, element)

        # Iterates through all positions of the 2D matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # if is a scalar it will raise an error when applying differentiate_finite
                if  isinstance(symbolic_matrix[i,j], numbers.Number):
                    matrix[i,j] = 0
                else:
                    matrix[i,j] = differentiate_finite(symbolic_matrix[i,j], symbol)
        return simplify(matrix)

    @classmethod
    def symbolic_matrix(cls, dim, asymmetrical=False, full_matrix=False):
        return cls._matrix_init(dim, asymmetrical=asymmetrical, full_matrix=full_matrix)

    @classmethod
    def C_lambda_mu(cls, model):
        def subs3D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': lmbda + (2*mu),
                    'C44': mu,
                    'C55': mu,
                    'C66': mu,
                    'C12': lmbda,
                    'C13': lmbda,
                    'C23': lmbda}

        def subs2D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': mu,
                    'C12': lmbda}

        matriz = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matriz.subs(subs)

        M.inv = cls._inverse_C_lam(model)
        return M

    @staticmethod
    def _inverse_C_lam(model):
        def subs3D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C44': 1/mu,
                    'C55': 1/mu,
                    'C66': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C13': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C23': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        def subs2D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        matrix = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @classmethod
    def C_vp_vs_rho(cls, model):
        def subs3D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vp*vp,
                    'C44': rho*vs*vs,
                    'C55': rho*vs*vs,
                    'C66': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs,
                    'C13': rho*vp*vp - 2*rho*vs*vs,
                    'C23': rho*vp*vp - 2*rho*vs*vs}

        def subs2D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = model.rho

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        M.inv = cls._inverse_C_vp_vs(model)
        return M

    @staticmethod
    def _inverse_C_vp_vs(model):
        def subs3D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C44': 1/(rho*vs*vs),
                    'C55': 1/(rho*vs*vs),
                    'C66': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C13': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C23': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        def subs2D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = model.rho

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @classmethod
    def C_Ip_Is_rho(cls, model):
        def subs3D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Ip*vp,
                    'C44': Is*vs,
                    'C55': Is*vs,
                    'C66': Is*vs,
                    'C12': Ip*vp - 2*Is*vs,
                    'C13': Ip*vp - 2*Is*vs,
                    'C23': Ip*vp - 2*Is*vs}

        def subs2D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Is*vs,
                    'C12': Ip*vp - 2*Is*vs}

        matrix = cls._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        Ip = model.Ip
        Is = model.Is

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        return M


class AuxiliaryMatrix(Matrix):

    def __getattr__(self, name):
        derivative_symbols = self._build_derivative_symbols()

        # if the desired attribute is in the list of derivative symbols,
        # generate its respective derivative matrix
        if name in derivative_symbols:
            return C_Matrix._generate_derivative(name, self)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _build_derivative_symbols(self):
        # generates the list of acceptable names indicating which derivative
        # matrix should be constructed
        parameters = C_Matrix.model.physical_parameters
        d_symb = ['d' + p for p in parameters]
        return d_symb
    

def D(self, shift=None):
    """
    Returns the result of matrix D applied over the TensorFunction.
    """
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")

    M = tensor(self) if self.shape[0] != self.shape[1] else self

    comps = []
    func = tens_func(self)
    for j, d in enumerate(self.space_dimensions):
        comps.append(sum([getattr(M[j, i], 'd%s' % d.name)
                         for i, d in enumerate(self.space_dimensions)]))
    return func._new(comps)


def S(self, shift=None):
    """
    Returns the result of transposed matrix D applied over the VectorFunction.
    """
    if not self.is_VectorValued:
        raise TypeError("The object must be a Vector object")

    derivs = ['d%s' % d.name for d in self.space_dimensions]

    comp = []
    comp.append(getattr(self[0], derivs[0]))
    comp.append(getattr(self[1], derivs[1]))
    if len(self.space_dimensions) == 3:
        comp.append(getattr(self[2], derivs[2]))
        comp.append(getattr(self[1], derivs[2]) + getattr(self[2], derivs[1]))
        comp.append(getattr(self[0], derivs[2]) + getattr(self[2], derivs[0]))
    comp.append(getattr(self[0], derivs[1]) + getattr(self[1], derivs[0]))

    func = tens_func(self)

    return func._new(comp)


def vec(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] != self.shape[1]:
        raise Exception("This object is already represented by its vector form.")

    order = ([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
             if len(self.space_dimensions) == 3 else [(0, 0), (1, 1), (0, 1)])
    comp = [self[o[0], o[1]] for o in order]
    func = tens_func(self)
    return func(comp)


def tensor(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] == self.shape[1]:
        raise Exception("This object is already represented by its tensor form.")

    ndim = len(self.space_dimensions)
    M = np.zeros((ndim, ndim), dtype=np.dtype(object))
    M[0, 0] = self[0]
    M[1, 1] = self[1]
    if len(self.space_dimensions) == 3:
        M[2, 2] = self[2]
        M[2, 1] = self[3]
        M[1, 2] = self[3]
        M[2, 0] = self[4]
        M[0, 2] = self[4]
    M[1, 0] = self[-1]
    M[0, 1] = self[-1]

    func = tens_func(self)
    return func._new(M)


def gather(a1, a2):

    expected_a1_types = [int, VectorFunction, VectorTimeFunction]
    expected_a2_types = [int, TensorFunction, TensorTimeFunction]

    if type(a1) not in expected_a1_types:
        raise ValueError("a1 must be a VectorFunction or a Integer")
    if type(a2) not in expected_a2_types:
        raise ValueError("a2 must be a TensorFunction or a Integer")
    if type(a1) is int and type(a2) is int:
        raise ValueError("Both a2 and a1 cannot be Integers simultaneously")

    if type(a1) is int:
        a1_m = Matrix([ones(len(a2.space_dimensions), 1)*a1])
    else:
        a1_m = Matrix(a1)

    if type(a2) is int:
        ndim = len(a1.space_dimensions)
        a2_m = Matrix([ones((3*ndim-3), 1)*a2])
    else:
        a2_m = Matrix(a2)

    if a1_m.cols > 1:
        a1_m = a1_m.T
    if a2_m.cols > 1:
        a2_m = a2_m.T

    return Matrix.vstack(a1_m, a2_m)
