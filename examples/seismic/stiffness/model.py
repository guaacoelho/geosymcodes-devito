from examples.seismic import SeismicModel
from examples.seismic.stiffness.utils import C_Matrix
from devito import Constant


class ElasticModel(SeismicModel):
    _known_parameters = SeismicModel._known_parameters + ['gamma', 'CC', 'Sw']

    def _initialize_physics(self, vp, space_order, **kwargs):
        # list o physical parameters there are mandatory for ElasticModel initialization
        mandatory_args = ('vs', 'rho')

        params = []

        # Check for the mandatory presence of physical attributes
        missing_args = [arg for arg in mandatory_args if arg not in kwargs]
        if missing_args:
            raise Exception(f"ElasticModel must receive {', '.join(missing_args)} as argument(s)")

        vs = kwargs.pop('vs')
        rho = kwargs.get('rho')

        self.rho = self._gen_phys_param(rho, 'rho', space_order)
        self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)*rho, 'lam', space_order,
                                        is_param=True)
        self.mu = self._gen_phys_param((vs**2) * rho, 'mu', space_order, is_param=True)
        self.vs = self._gen_phys_param(vs, 'vs', space_order)
        self.vp = self._gen_phys_param(vp, 'vp', space_order)

        self.Ip = self._gen_phys_param(vp*rho, 'Ip', space_order, is_param=True)
        self.Is = self._gen_phys_param(vs*rho, 'Is', space_order, is_param=True)

        E = (rho*vs**2)*((3*vp**2 - 4*vs**2)/(vp**2 - vs**2))
        nu = (vp**2 - 2*vs**2)/(2*(vp**2 - vs**2))
        self.E = self._gen_phys_param(E, 'E', space_order, is_param=True)
        self.nu = self._gen_phys_param(nu, 'nu', space_order, is_param=True)

        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

        self._initialize_C_arguments(space_order, **kwargs)

    def _initialize_C_arguments(self, space_order, **kwargs):
        symbs_C = C_Matrix.symbolic_matrix(self.dim, asymmetrical=True, full_matrix=True).free_symbols

        missing_params = [s.name for s in symbs_C if s.name not in kwargs]
        if missing_params and len(missing_params) != len(symbs_C):
            raise ValueError(
                f"If you define a value for matrix C, you must define values "
                f"for all its elements. Missing: {', '.join(missing_params)}"
            )

        if not missing_params:
            for s in symbs_C:
                new_parameter = self._gen_phys_param(kwargs.get(s.name), s.name,
                                                     space_order, is_param=True)
                setattr(self, s.name, new_parameter)
