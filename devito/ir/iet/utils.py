from devito.ir.iet import FindSections, FindSymbols, Call, Conditional, FindNodes, Transformer, Iteration, Section
from devito.symbolics import Keyword, Macro, CondEq, String, Null
from devito.tools import filter_ordered
from devito.types import Global, SpaceDimension, TimeDimension
from sympy import Or

__all__ = ['filter_iterations', 'retrieve_iteration_tree', 'derive_parameters',
           'maybe_alias', 'array_alloc_check', 'get_first_space_dim_index', 'update_iet', 'get_compress_mode_function']


class IterationTree(tuple):

    """
    Represent a sequence of nested Iterations.
    """

    @property
    def root(self):
        return self[0] if self else None

    @property
    def inner(self):
        return self[-1] if self else None

    @property
    def dimensions(self):
        return [i.dim for i in self]

    def __repr__(self):
        return "IterationTree%s" % super(IterationTree, self).__repr__()

    def __getitem__(self, key):
        ret = super(IterationTree, self).__getitem__(key)
        return IterationTree(ret) if isinstance(key, slice) else ret


def retrieve_iteration_tree(node, mode='normal'):
    """
    A list with all Iteration sub-trees within an IET.

    Examples
    --------
    Given the Iteration tree:

        .. code-block:: c

           Iteration i
             expr0
             Iteration j
               Iteration k
                 expr1
             Iteration p
               expr2

    Return the list: ::

        [(Iteration i, Iteration j, Iteration k), (Iteration i, Iteration p)]

    Parameters
    ----------
    iet : Node
        The searched Iteration/Expression tree.
    mode : str, optional
        - ``normal``
        - ``superset``: Iteration trees that are subset of larger iteration trees
                        are dropped.
    """
    assert mode in ('normal', 'superset')

    trees = [IterationTree(i) for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        found = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            found.append(i)
        return found


def filter_iterations(tree, key=lambda i: i):
    """
    Return the first sub-sequence of consecutive Iterations such that
    ``key(iteration)`` is True.
    """
    filtered = []
    for i in tree:
        if key(i):
            filtered.append(i)
        elif len(filtered) > 0:
            break
    return filtered


def derive_parameters(iet, drop_locals=False):
    """
    Derive all input parameters (function call arguments) from an IET
    by collecting all symbols not defined in the tree itself.
    """
    # Extract all candidate parameters
    candidates = FindSymbols().visit(iet)

    # Symbols, Objects, etc, become input parameters as well
    basics = FindSymbols('basics').visit(iet)
    candidates.extend(i.function for i in basics)
    
    # Filter off duplicates (e.g., `x_size` is extracted by both calls to FindSymbols)
    candidates = filter_ordered(candidates)

    # Filter off symbols which are defined somewhere within `iet`
    defines = [s.name for s in FindSymbols('defines').visit(iet)]
    parameters = [s for s in candidates if (s.name not in defines and not s.ignoreDefinition)]

    # Drop globally-visible objects
    parameters = [p for p in parameters
                  if not isinstance(p, (Global, Keyword, Macro))]
    
    # Drop (to be) locally declared objects as well as global objects
    parameters = [p for p in parameters
                  if not (p._mem_internal_eager or p._mem_constant)]
    
    # Maybe filter out all other compiler-generated objects
    if drop_locals:
        parameters = [p for p in parameters if not (p.is_ArrayBasic or p.is_LocalObject)]

    return parameters


def maybe_alias(obj, candidate):
    """
    True if `candidate` can act as an alias for `obj`, False otherwise.
    """
    if obj is candidate:
        return True

    # Names are unique throughout compilation, so this is another case we can handle
    # straightforwardly. It might happen that we have an alias used in a subroutine
    # with different type qualifiers (e.g., const vs not const, volatile vs not
    # volatile), but if the names match, they definitely represent the same
    # logical object
    if obj.name == candidate.name:
        return True

    if obj.is_AbstractFunction:
        if not candidate.is_AbstractFunction:
            # Obv
            return False

        # E.g. TimeFunction vs SparseFunction -> False
        if type(obj).__base__ is not type(candidate).__base__:
            return False

        # TODO: At some point we may need to introduce some logic here, but we'll
        # also need to introduce something like __eq_weak__ that compares most of
        # the __rkwargs__ except for e.g. the name

    return False

def array_alloc_check(arrays):
    """
    Checks wether malloc worked for array allocation.

    Args:
        array (Array): array (files or counters)

    Returns:
        Conditional: condition to handle allocated array
    """
    
    eqs = []
    for arr in arrays:
        eqs.append(CondEq(arr, Macro('NULL')))
    
    ors = Or(*eqs)
    
    pstring = String("\"Error to alloc\"")
    printfCall = Call(name="printf", arguments=pstring)
    exitCall = Call(name="exit", arguments=1)
    return Conditional(ors, [printfCall, exitCall])

def get_first_space_dim_index(dimensions):
    """
    This method returns the index of the first space dimension of the Function.

    Args:
        dimensions (tuple): dimensions

    Returns:
        int: index
    """
    
    first_space_dim_index = 0
    for dim in dimensions:
        if isinstance(dim, SpaceDimension):
            break
        else:
            first_space_dim_index += 1
    
    return first_space_dim_index

def update_iet(iet_body, temp_name, ooc_section):
    """
    This function substitute a temp section with definitive section.

    Args:
        iet_body (List): IET body noes
        temp_name (string): name of section
        ooc_section (Section): Read/Decompress or Write/Compress section
    """
    
    sections = FindNodes(Section).visit(iet_body)
    temp_sec = next((section for section in sections if section.name == temp_name), None)
    mapper={temp_sec: ooc_section}

    timeIndex = next((i for i, node in enumerate(iet_body) if isinstance(node, Iteration) and isinstance(node.dim, TimeDimension)), None)
    transformedIet = Transformer(mapper).visit(iet_body[timeIndex])
    iet_body[timeIndex] = transformedIet 

def get_compress_mode_function(compress_config, zfp, field, Type):
    """_summary_

    Args:
        compress_config (CompressionConfig): object with compress settings
        zfp (zfp_stream): _description_
        field (_type_): _description_
        Type (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    
    arguments = [zfp]
    if compress_config.mode == "set_rate":
        arguments += [compress_config.rate, Type, Call(name="zfp_field_dimensionality", arguments=[field]), String(r"zfp_false")]
    elif compress_config.mode == "set_accuracy" or compress_config.mode == "set_precision":
        arguments.append(compress_config.value)     
        
    return Call(name="zfp_stream_"+compress_config.mode, arguments=arguments)