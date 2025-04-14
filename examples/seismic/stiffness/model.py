from examples.seismic import SeismicModel
from examples.seismic.stiffness.utils import C_Matrix


class ElasticModel(SeismicModel):
    _known_parameters = SeismicModel._known_parameters + ['gamma']

    def _initialize_physics(self, vp, space_order, **kwargs):

        params = []
        # Buoyancy
        rho = kwargs.get('rho', 1)
        self.rho = self._gen_phys_param(rho, 'rho', space_order)

        # Initialize elastic with Lame parametrization
        try:
            vs = kwargs.pop('vs')
        except:
            raise Exception("ElasticModel must receive 'vs' as an argument")

        self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)*rho, 'lam', space_order,
                                        is_param=True)
        self.mu = self._gen_phys_param((vs**2) * rho, 'mu', space_order, is_param=True)
        self.vs = self._gen_phys_param(vs, 'vs', space_order)
        self.vp = self._gen_phys_param(vp, 'vp', space_order)

        self.Ip = self._gen_phys_param(vp*rho, 'Ip', space_order, is_param=True)
        self.Is = self._gen_phys_param(vs*rho, 'Is', space_order, is_param=True)

        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

        self._initialize_C_arguments(space_order, **kwargs)

    def _initialize_C_arguments(self, space_order, **kwargs):
        symbs_C = C_Matrix.symbolic_matrix(self.dim).free_symbols
        for s in symbs_C:
            if s.name in kwargs:
                new_parameter = self._gen_phys_param(kwargs.get(s.name), s.name,
                                                     space_order, is_param=True)
                setattr(self, s.name, new_parameter)

                # Mark if the C_Matrix parameters have been initialized
                if not hasattr(self, "has_C_params"):
                    self.has_C_params = True
