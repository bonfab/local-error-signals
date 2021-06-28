'''
@original code at: https://github.com/janrain/lazydict

changed: self.values to self.vals as it overshadowed the method values()
'''

from collections.abc import MutableMapping
from dask.utils import SerializableLock as RLock
from inspect import getfullargspec
from copy import copy
from pathlib import Path

def get_version():
    VERSION = (     # SEMANTIC
        1,          # major
        0,          # minor
        0,          # patch
        'beta.2',   # pre-release
        None        # build metadata
    )

    version = "%i.%i.%i" % (VERSION[0], VERSION[1], VERSION[2])
    if VERSION[3]:
        version += "-%s" % VERSION[3]
    if VERSION[4]:
        version += "+%s" % VERSION[4]
    return version

CONSTANT = frozenset(['evaluating', 'evaluated', 'error'])

class LazyDictionaryError(Exception):
    pass

class CircularReferenceError(LazyDictionaryError):
    pass

class ConstantRedefinitionError(LazyDictionaryError):
    pass

class LazyDictionary(MutableMapping):
    def __init__(self, vals={}):
        # self.lock = RLock()
        self.vals = copy(vals)
        self.states = {}
        for key in self.vals:
            self.states[key] = 'defined'

    def __len__(self):
        return len(self.vals)

    def __iter__(self):
        return iter(self.vals)

    def __getitem__(self, key):
        # with self.lock:
        if key in self.states:
            if self.states[key] == 'evaluating':
                raise CircularReferenceError('value of "%s" depends on itself' % key)
            elif self.states[key] == 'error':
                raise self.vals[key]
            elif self.states[key] == 'defined':
                value = self.vals[key]
                if callable(value):
                    argspec = getfullargspec(value)
                    if len(argspec.args) == 0:
                        self.states[key] = 'evaluating'
                        try:
                            self.vals[key] = value()
                        except Exception as ex:
                            self.vals[key] = ex
                            self.states[key] = 'error'
                            raise ex
                    elif len(argspec.args) == 1:
                        self.states[key] = 'evaluating'
                        try:
                            self.vals[key] = value(self)
                        except Exception as ex:
                            self.vals[key] = ex
                            self.states[key] = 'error'
                            raise ex
                self.states[key] = 'evaluated'
        return self.vals[key]

    def __contains__(self, key):
        return key in self.vals

    def __setitem__(self, key, value):
        # with self.lock:
        if self.states.get(key) in CONSTANT:
            raise ConstantRedefinitionError('"%s" is immutable' % key)
        self.vals[key] = value
        self.states[key] = 'defined'

    def __delitem__(self, key):
        # with self.lock:
        if self.states.get(key) in CONSTANT:
            raise ConstantRedefinitionError('"%s" is immutable' % key)
        del self.vals[key]
        del self.states[key]

    def __str__(self):
        return str(self.vals)

    def __repr__(self):
        return "LazyDictionary({0})".format(repr(self.vals))

class MutableLazyDictionary(LazyDictionary):
    def __setitem__(self, key, value):
        with self.lock:
            self.vals[key] = value
            self.states[key] = 'defined'

    def __delitem__(self, key):
        with self.lock:
            del self.vals[key]
            del self.states[key]


# class CallableDict(dict):
#     def __getitem__(self, key):
#         val = super().__getitem__(key)
#         if callable(val):
#             return val()
#         return val

#     def __repr__(self):
#         return f"CallableDict({super().__repr__()})"


class CallableDict(MutableMapping):
    def __init__(self, vals={}):
        self.vals = copy(vals)

    def __len__(self):
        return len(self.vals)

    def __iter__(self):
        return iter(self.vals)

    def __getitem__(self, key):
        val = self.vals[key]
        if callable(val):
            return val()
        return val

    def __contains__(self, key):
        return key in self.vals

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __delitem__(self, key):
        del self.vals[key]

    def __str__(self):
        return str(self.vals)

    def __repr__(self):
        return f"CallableDict({repr(self.vals)})"

    def update(self, other):
        if isinstance(other, type(self)):
            iterable = other.vals
        else:
            iterable = other

        for key, val in iterable.items():
            self.vals[key] = val


class RootPathLazyDictionary(CallableDict):
    def __init__(self, vals={}, root_path=None):
        super().__init__(vals=vals)
        self.root_path = root_path

    def __setitem__(self, key, value):
        if key == 'root_path':
            self.root_path = value
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if key == 'root_path':
            return self.root_path
        val = self.vals[key]
        if callable(val):
            argspec = getfullargspec(val)
            if len(argspec.args) == 0:
                return val()
            if len(argspec.args) == 1:
                return val(self)
            raise ValueError(f"Can only pass the whole dict. Wants: {argspec}")
        return val

    def update(self, other):
        if isinstance(other, type(self)):
            assert (other.root_path == self.root_path) or (
                self.root_path is None
            ), f"root_path doesn't match, self: {self.root_path}, other: {other.root_path}"

            self.root_path = other.root_path
        
        super().update(other)






class CallWithPath():
    def __init__(self, fn, path, use_root_from_dict='root_path'):
        self.fn = fn
        self.path = path
        self.use_root_from_dict = use_root_from_dict

    def __call__(self, *args, **kwargs):
        if self.use_root_from_dict:
            root_path = args[0][self.use_root_from_dict]
        else:
            root_path = Path()
        return self.fn(root_path/self.path)
