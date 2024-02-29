from ctypes import c_double, c_void_p, c_int, Structure, c_uint64, c_int64, c_float, c_size_t, c_byte

import numpy as np
from sympy.core.core import ordering_of_classes
from sympy.codegen.ast import SignedIntType

from devito.types import CompositeObject, Indexed, Symbol
from devito.types.basic import IndexedData
from devito.tools import Pickable, as_tuple

__all__ = ['Timer', 'Pointer', 'VolatileInt', 'FIndexed', 'Wildcard',
           'Global', 'Hyperplane', 'Indirection', 'Temp', 'Jump', 'FILE', 
           'off_t', 'size_t', 'zfp_type', 'zfp_field', 'bitstream', 'zfp_stream']


class Timer(CompositeObject):

    __rargs__ = ('name', 'sections')

    def __init__(self, name, sections, **kwargs):
        super().__init__(name, 'profiler', [(i, c_double) for i in sections], **kwargs)

    def reset(self):
        for i in self.fields:
            setattr(self.value._obj, i, 0.0)
        return self.value

    @property
    def total(self):
        return sum(getattr(self.value._obj, i) for i in self.fields)

    @property
    def sections(self):
        return self.fields

    def _arg_values(self, **kwargs):
        values = super()._arg_values(**kwargs)

        # Reset timer
        for i in self.fields:
            setattr(values[self.name]._obj, i, 0.0)

        return values


class VolatileInt(Symbol):
    is_volatile = True


class Wildcard(Symbol):

    """
    A special Symbol used by the compiler to generate ad-hoc code
    (e.g. to work around known bugs in jit-compilers).
    """

    pass


class FIndexed(Indexed, Pickable):

    """
    A flatten Indexed with functional (primary) and indexed (secondary) representations.

    Examples
    --------
    Consider the Indexed `u[x, y]`. The corresponding FIndexed's functional representation
    is `u(x, y)`. This is a multidimensional representation, just like any other Indexed.
    The corresponding indexed (secondary) represenation is instead flatten, that is
    `uX[x*ny + y]`, where `X` is a string provided by the caller.
    """

    __rargs__ = ('base', '*indices')
    __rkwargs__ = ('strides',)

    def __new__(cls, base, *args, strides=None):
        obj = super().__new__(cls, base, *args)
        obj.strides = as_tuple(strides)

        return obj

    @classmethod
    def from_indexed(cls, indexed, pname, strides=None):
        label = Symbol(name=pname, dtype=indexed.dtype)
        base = IndexedData(label, None, function=indexed.function)
        return FIndexed(base, *indexed.indices, strides=strides)

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(str(i) for i in self.indices))

    __str__ = __repr__

    def _hashable_content(self):
        return super()._hashable_content() + (self.strides,)

    func = Pickable._rebuild

    @property
    def name(self):
        return self.function.name

    @property
    def pname(self):
        return self.base.name

    @property
    def free_symbols(self):
        # The functional representation of the FIndexed "hides" the strides, which
        # are however actual free symbols of the object, since they contribute to
        # the address calculation just like all other free_symbols
        return (super().free_symbols |
                set().union(*[i.free_symbols for i in self.strides]))

    func = Pickable._rebuild

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Global(Symbol):

    """
    A special Symbol representing global variables.
    """

    pass


class Hyperplane(tuple):

    """
    A collection of Dimensions defining an hyperplane.
    """

    @property
    def _defines(self):
        return frozenset().union(*[i._defines for i in self])


class Pointer(Symbol):

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', c_void_p)

    @property
    def _C_ctype(self):
        # `dtype` is a ctypes-derived type!
        return self.dtype


class Indirection(Symbol):

    """
    An Indirection is a Symbol that holds a value used to indirectly access
    an Indexed.

    Examples
    --------
    Below an Indirection, `ofs`, used to access an array `a`.

        ofs = offsets[time + 1]
        v = a[ofs]
    """

    __rkwargs__ = Symbol.__rkwargs__ + ('mapped',)

    def __new__(cls, name=None, mapped=None, dtype=np.uint64, is_const=True,
                **kwargs):
        obj = super().__new__(cls, name=name, dtype=dtype, is_const=is_const,
                              **kwargs)
        obj.mapped = mapped

        return obj


class Temp(Symbol):

    """
    A Temp is a Symbol used by compiler passes to store locally-constructed
    temporary expressions.
    """

    # Just make sure the SymPy args ordering is the same regardless of whether
    # the arguments are Symbols or Temps
    ordering_of_classes.insert(ordering_of_classes.index('Symbol') + 1, 'Temp')


class Jump(object):

    """
    Mixin class for symbolic objects representing jumps in the control flow,
    such as return and break statements.
    """

    pass


class FILE(Structure):
    """
    Class representing the FILE structure type in C/C++.
    """
    _fields_ = [("FILE", c_int)]
    
    
class off_t(c_int64):
    
    """
    Class representing the off_t type in C/C++
    """

    pass

class size_t(c_uint64):
    
    """
    Class representing the size_t type in C/C++
    """

    pass

### Compression specific classes ###
class zfp_type(c_int):
    
    # NOTE: I don't know how to specify an unspecified type
    # NOTE: Maybe the more appropriate solution is making zfp_type subclass of Structure and develop it with __fields__,
    # but devito does not translate it the way I want
    
    """
    Class representing:
    
    typedef enum {
        zfp_type_none   = 0, // unspecified type
        zfp_type_int32  = 1, // 32-bit signed integer
        zfp_type_int64  = 2, // 64-bit signed integer
        zfp_type_float  = 3, // single precision floating point
        zfp_type_double = 4  // double precision floating point
    } zfp_type;
    """    
    # _fields_ = [
    #     # ("zfp_type_int32", c_int32),
    #     # ("zfp_type_int64", c_int64),
    #     # ("zfp_type_float", c_float),
    #     # ("zfp_type_double", c_double)
    # ]
    pass
    
class zfp_field(c_int):
    
    # NOTE: Maybe the more appropriate solution is making zfp_field subclass of Structure and develop it with __fields__,
    # but devito does not translate it the way I want
    
    """
    Class representing:
    
    typedef struct {
        zfp_type type;            // scalar type (e.g., int32, double)
        size_t nx, ny, nz, nw;    // sizes (zero for unused dimensions)
        ptrdiff_t sx, sy, sz, sw; // strides (zero for contiguous array a[nw][nz][ny][nx])
        void* data;               // pointer to array data
    } zfp_field;
    """
    
    # _fields_ = [
    #     ("zfp_field", c_float),
    #     ("type", zfp_type),
    #     ("nx", c_size_t), ("ny", c_size_t), ("nz", c_size_t),("nw", c_size_t),
    #     ("sx", c_int), ("sy", c_int), ("sz", c_int),("sw", c_int),
    #     ("data", c_void_p)
    # ]
    pass

class bitstream(c_int):
    
    # NOTE: Maybe the more appropriate solution is making bitstream subclass of Structure and develop it with __fields__,
    # but devito does not translate it the way I want
    
    """
    Class representing:
    
    struct bitstream {
        bitstream_count bits;  // number of buffered bits (0 <= bits < word size)
        bitstream_word buffer; // incoming/outgoing bits (buffer < 2^bits)
        bitstream_word* ptr;   // pointer to next word to be read/written
        bitstream_word* begin; // beginning of stream
        bitstream_word* end;   // end of stream (not enforced)
        size_t mask;           // one less the block size in number of words (if BIT_STREAM_STRIDED)
        ptrdiff_t delta;       // number of words between consecutive blocks (if BIT_STREAM_STRIDED)
    };
    """
    
    # _fields_ = [
    #     ("minbits", c_uint),
    #     ("maxbits", c_uint), 
    #     ("maxprec", c_uint), 
    #     ("minexp", c_int)
    # ]
    pass

class zfp_stream(c_int):
    
    # NOTE: Maybe the more appropriate solution is making zfp_stream subclass of Structure and develop it with __fields__,
    # but devito does not translate it the way I want
    
    """
    Class representing:
    
    typedef struct {
        uint minbits;       // minimum number of bits to store per block
        uint maxbits;       // maximum number of bits to store per block
        uint maxprec;       // maximum number of bit planes to store
        int minexp;         // minimum floating point bit plane number to store
        bitstream* stream;  // compressed bit stream
        zfp_execution exec; // execution policy and parameters
    } zfp_stream;
    """
    
    # _fields_ = [
    #     ("minbits", c_uint),
    #     ("maxbits", c_uint), 
    #     ("maxprec", c_uint), 
    #     ("minexp", c_int)
    # ]    
    pass
    

