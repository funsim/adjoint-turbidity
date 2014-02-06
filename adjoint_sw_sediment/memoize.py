from dolfin import *
from dolfin_adjoint import *
import pickle
import signal
import os

def to_tuple(obj):
    if hasattr(obj, '__iter__'):
        return tuple([to_tuple(o) for o in obj])
    else:
        return obj

class MemoizeMutable:
    ''' Implements a memoization function to avoid duplicated functional (derivative) evaluations '''

    def get_key(self, args, kwds):
        h1 = to_tuple(args)
        h2 = to_tuple(kwds.items())
        h = tuple([h1, h2])
        # Often useful to have a explicit 
        # turbine parameter -> functional value mapping,
        # i.e. no hashing on the key
        if self.hash_keys:
            h = hash(h)  
        return h

    def __init__(self, fn, hash_keys=False):
        ''' sigint_save: Create a checkpoint file in case a sigint signal is received. '''
        self.fn = fn
        self.memo = {}
        self.hash_keys = hash_keys

    def __call__(self, *args, **kwds):
        h = self.get_key(args, kwds)

        if h not in self.memo:
            self.memo[h] = self.fn(*args, **kwds)
        else:
            print ("Use checkpoint value.")

        return self.memo[h]

    def has_cache(self, *args, **kwds):
        h = self.get_key(args, kwds)
        return h in self.memo

    # Insert a function value into the cache manually.
    def __add__(self, value, *args, **kwds):
        h = self.get_key(args, kwds)
        self.memo[h] = value

    def save_checkpoint(self, filename):
        def sig_save(sig, stack):
            print "Received signal %i. Writing final checkpoint to disk before exiting..." % sig
            pickle.dump(self.memo, open(filename, "wb"))
            print "Checkpoint writing finished. Bye."
            os._exit(sig)

        # Make sure we save successfully, even if the user sends a signal
        print "Save checkpoint."
        old_handler = signal.signal(signal.SIGINT, sig_save)
        pickle.dump(self.memo, open(filename, "w"))
        signal.signal(signal.SIGINT, old_handler)

    def load_checkpoint(self, filename):
        try:
            self.memo = pickle.load(open(filename, "r"))
        except IOError:
            info_red("Warning: Checkpoint file '%s' not found." % filename)
