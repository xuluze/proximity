# needs https://github.com/sagemath/sage/pull/36031

#########################################################################################################
# the code below is an adaptation of the code for delta-modular matrices classification
# by Matthias Schymura and Gennadiy Averkov
# https://github.com/mschymura/delta-classification
# which is an adaptation of the code for mixed volume classification by Christopher Borger
# https://github.com/christopherborger/mixed_volume_classification/blob/master/volume_classification.sage
#########################################################################################################

import functools
import itertools
import logging
import os.path
import sys

from collections import defaultdict

from sage.geometry.palp_normal_form import _palp_PM_max, _palp_canonical_order
from sage.geometry.polyhedron.parent import Polyhedra
from sage.misc.lazy_attribute import lazy_attribute

# Using the logging package one can conveniently turn off and on the auxiliary messages

logging.basicConfig(format='%(message)s',stream=sys.stdout,level=logging.INFO)
# After modifying the level from, say, logging.INFO to logging.WARNING , the change will come into force only after _restarting the sage session_ and reloading

# Sandwich is a pair of centrally symmetric lattice polytopes A,B with A being a subset of B.
# For the sake of efficiency, A also comes with its "symmetry-broken" part halfA such that A = halfA \cup -halfA \cup {0}.
# The gap of a sandwich A,B is the difference |B \cap Z^d| - |A \cap Z^d| of the number of integer points in B and A.

# that's the template for names of files, in which we store polytopes
FILE_NAME_DELTA = 'data/dim_%d_delta_%d.txt'
FILE_NAME_DELTA_EXTR = 'data/dim_%d_delta_%d_extremal.txt'
FILE_NAME_DELTA_MAX = 'data/dim_%d_delta_%d_maximal.txt'
FILE_NAME_DELTA_CONE = 'data/dim_%d_delta_%d_trivial_cone.txt'


class Sandwich_base:

    # Methods needed for storage in SandwichStorage:

    def polyhedra_parent(self):
        return self._A.parent()

    @abstract_method
    def __repr__(self):
        pass

    @abstract_method
    def plot(self):
        pass

    @lazy_attribute
    @abstract_method
    def _halfA(self):
        pass

    @lazy_attribute
    def _A_integral_points(self):
        return self._A.integral_points()

    def A_integral_points(self):
        return self._A_integral_points

    def A_integral_points_count(self):
        return len(self.A_integral_points())

    @cached_method
    @abstract_method
    def gap(self):
        pass

    @cached_method
    def _key_func_A_dimensions(self):
        return (self._A.n_facets(), self._A.n_vertices())

    @lazy_attribute
    def _A_vertex_facet_pairing_matrix(self):
        return self._A.slack_matrix().transpose()

    @staticmethod
    def _row_sums(matrix):
        return Partition(sorted(sum(matrix.columns()), reverse=True))

    @staticmethod
    def _row_power_sums(matrix, powers):
        return tuple(Partition(sorted((sum(x**k for x in row) for row in matrix.rows()),
                                      reverse=True))
                     for k in powers)

    @staticmethod
    def _column_sums(matrix):
        return Partition(sorted(sum(matrix.rows()), reverse=True))

    @staticmethod
    def _column_power_sums(matrix, powers):
        return tuple(Partition(sorted((sum(x**k for x in column) for column in matrix.columns()),
                                      reverse=True))
                     for k in powers)

    @staticmethod
    def _row_maxes(matrix):
        return Partition(sorted((max(x for x in row) for row in matrix.rows()),
                                reverse=True))

    @staticmethod
    def _column_maxes(matrix):
        return Partition(sorted((max(x for x in column) for column in matrix.columns()),
                                reverse=True))

    @staticmethod
    def _row_and_column_sums_and_maxes(matrix):
        return (Sandwich_base._row_sums(matrix), Sandwich_base._column_sums(matrix),
                Sandwich_base._row_maxes(matrix), Sandwich_base._column_maxes(matrix))

    @staticmethod
    def _row_and_column_power_sums(matrix, powers):
        return (Sandwich_base._row_power_sums(matrix, powers), Sandwich_base._column_power_sums(matrix, powers))

    @cached_method
    def _key_func_A_partitions(self):
        r"""
        Invariants: degree-1 symmetric functions, max symmetric functions of rows and columns
        """
        return Sandwich_base._row_and_column_sums_and_maxes(self._A_vertex_facet_pairing_matrix)

    @cached_method(do_pickle=True)
    def _key_func_A_palp_native_normal_form(self):
        return tuple(self._A.normal_form())

    @abstract_method
    def key_funcs(self):
        r"""
        Return a sequence of functions returning components of the key
        """
        pass

    @abstract_method
    def key_costs(self):
        r"""
        Return a sequence of (integer) costs of the components of the key
        """
        pass

    # Optional
    @abstract_method
    def noninvariant_keys(self):
        r"""
        Return a sequence of keys to check if two sandwich are the same (not just invariant under unimodular transformation)
        """
        pass

    # Default implementations

    def __len__(self):
        return len(self.key_funcs())

    def __getitem__(self, i):
        r"""
        Return the components of the key for the trie
        """
        return self.key_funcs()[i]()

    def item_cost(self, i):
        try:
            if self.key_funcs()[i].is_in_cache():
                return 0
        except AttributeError:
            pass
        return self.key_costs()[i]

    def __eq_noninvariant__(self, other):
        if self is other:
            return True
        return self.noninvariant_keys() == other.noninvariant_keys()

    def __eq__(self, other):
        r"""
        Assumes that ``self`` and ``other`` are the same when all of their keys are the same
        """
        # First check fast non-invariant
        if self.__eq_noninvariant__(other):
            return True
        return all(s == o for s, o in zip(self, other))


class Sandwich(Sandwich_base):
    r"""
    A sandwich of lattice polytopes, equipped with a sequence of invariants as keys

    To profile sequences of invariants::

        sage: %prun -s ncalls -l _key_func_ delta_classification(3, 1, False)  # not tested

    EXAMPLES::

        sage: S = Sandwich(Polyhedron([[2, 3], [4, 5], [6, 7]]), Polyhedron([[0, 0], [0, 7], [7, 7], [7, 0]]))
        sage: list(S)

    """
    def __init__(self, A, B, *, A_integral_points=None, B_integral_points=None, order=None):
        if isinstance(A, (tuple, list)):
            # Assume it's [halfA, A] where A is a Polyhedron
            self._A = A[1]
            self._halfA = A[0]
        else:
            self._A = A

        if A_integral_points is not None:
            self._A_integral_points = A_integral_points

        if B_integral_points is not None:
            self._B_integral_points = B_integral_points
        elif isinstance(B, (tuple, list)):
            self._B_integral_points = B
        else:
            self._B = B

        self._order = order

    def __repr__(self):
        if self.gap():
            return f"Sandwich conv({sorted(self._A.vertices_list())}) ⊆ conv({sorted(self._B.vertices_list())}) with gap {self.gap()}"
        return f"Polytope conv({sorted(self._A.vertices_list())})"

    def plot(self):
        return self._B.plot(alpha=.3, polygon='yellow') + self._A.plot(alpha=.3, polygon='red')

    @lazy_attribute
    def _halfA(self):
        A = self._A
        m = A.ambient_dim()
        return break_symmetry(A, m)

    @lazy_attribute
    def _B_integral_points(self):
        return self._B.integral_points()

    @lazy_attribute
    def _B(self):
        return self.polyhedra_parent()([self._B_integral_points, [], []], None)

    def B_integral_points(self):
        r"""
        Return a tuple of immutable vectors
        """
        return self._B_integral_points

    def B_integral_points_count(self):
        return len(self.B_integral_points())

    def B_minus_A_integral_points(self):
        return tuple(v for v in self.B_integral_points() if v not in self.A_integral_points())

    @cached_method
    def gap(self):
        return self.B_integral_points_count() - self.A_integral_points_count()

    @cached_method
    def _key_func_dimensions(self):
        return (self._A.n_facets(), self._A.n_vertices(), self._B.n_facets(), self._B.n_vertices())

    @lazy_attribute
    def _B_vertex_facet_pairing_matrix(self):
        return self._B.slack_matrix().transpose()

    @cached_method
    def _key_func_B_partitions(self):
        r"""
        Invariants: degree-1 symmetric functions, max symmetric functions of rows and columns
        """
        row_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.columns()), reverse=True))
        column_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.rows()), reverse=True))
        column_maxes = Partition(sorted((max(x for x in column)
                                        for column in self._B_vertex_facet_pairing_matrix.columns()),
                                        reverse=True))
        row_maxes = Partition(sorted((max(x for x in row)
                                      for row in self._B_vertex_facet_pairing_matrix.rows()),
                                     reverse=True))

        #print(row_sums, column_sums)
        return row_sums, column_sums, row_maxes, column_maxes

    @lazy_attribute
    def _A_vertex_B_facet_pairing_matrix(self):

        Vrep_matrix = matrix(ZZ, self._A.Vrepresentation())
        Hrep_matrix = matrix(ZZ, self._B.Hrepresentation())

        # Getting homogeneous coordinates of the Vrepresentation.
        hom_helper = matrix(ZZ, [1 if v.is_vertex() else 0 for v in self._A.Vrepresentation()])
        hom_Vrep = hom_helper.stack(Vrep_matrix.transpose())

        PM = Hrep_matrix * hom_Vrep
        PM.set_immutable()
        return PM

    @cached_method
    def _key_func_A_vertex_B_facet_partitions(self):
        #return Sandwich._row_and_column_sums_and_maxes(self._A_vertex_B_facet_pairing_matrix)
        return Sandwich._row_and_column_power_sums(self._A_vertex_B_facet_pairing_matrix, range(1, 3))

    @cached_method(do_pickle=True)
    def _key_func_B_permutation_normal_form(self):
        PNF = self._B_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        PNF.set_immutable()
        return PNF

    @cached_method(do_pickle=True)
    def _key_func_A_vertex_B_facet_permutation_normal_form(self):
        PNF = self._A_vertex_B_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        PNF.set_immutable()
        return PNF

    @lazy_attribute
    def _LLP(self):
        return layered_polytope_from_sandwich((None, self._A), self._B)

    @lazy_attribute
    def _LLP_vertex_facet_pairing_matrix(self):
        return self._LLP.slack_matrix().transpose()

    @lazy_attribute
    def _LLP_PM_max_and_permutations(self):
        PM_max, permutations = _palp_PM_max(self._LLP_vertex_facet_pairing_matrix, check=True)
        PM_max.set_immutable()
        return PM_max, permutations

    @cached_method(do_pickle=True)
    def _key_func_LLP_permutation_normal_form(self):
        "FIXME: This is apparently NOT a normal form"
        #return self._LLP.normal_form(algorithm='palp')  # fastest of all, but crashes for dim > 3.
        #PNF = self._LLP_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        #PNF = self._LLP._palp_PM_max(check=False)       # slower (before https://github.com/sagemath/sage/pull/35997), much faster (after)

        #PNF.set_immutable()
        PNF = self._LLP_PM_max_and_permutations[0]       # same as above, but stores permutations for use by _key_func_LLP_palp_native_normal_form below
        return PNF

    @cached_method(do_pickle=True)
    def _key_func_LLP_palp_native_normal_form(self):
        PM_max, permutations = self._LLP_PM_max_and_permutations
        return tuple(_palp_canonical_order(self._LLP.vertices(), PM_max, permutations)[0])

    def key_funcs(self):
        if self.gap():
            return (self._key_func_dimensions,
                    self._key_func_A_vertex_B_facet_partitions)
        return (self._key_func_A_dimensions,
                # self._key_func_A_partitions,
                #self._key_func_B_partitions,
                self._key_func_A_vertex_B_facet_partitions,
                #self._key_func_B_permutation_normal_form,
                #self._key_func_A_vertex_B_facet_permutation_normal_form,
                #self._key_func_LLP_permutation_normal_form,
                #self._key_func_LLP_palp_native_normal_form,   # Actually no need to construct LLP b/c gap=0
                self._key_func_A_palp_native_normal_form)

    @staticmethod
    def key_costs():
        return (0,
                1,
                50)

    @cached_method
    def noninvariant_keys(self):
        return (self._halfA,
                tuple(sorted(tuple(int(x) for x in v) for v in self._B.vertices())))


class Sandwich_with_order(Sandwich_base):
    r"""
    A sandwich of lattice polytopes, equipped with a sequence of invariants as keys,
    where the upper bound is maintained as a set of possible points with an order to add

    To profile sequences of invariants::

        sage: %prun -s ncalls -l _key_func_ delta_classification(3, 1, False)  # not tested

    EXAMPLES::

        sage: S = Sandwich(Polyhedron([[2, 3], [4, 5], [6, 7]]), Polyhedron([[0, 0], [0, 7], [7, 7], [7, 0]]))
        sage: list(S)

    """
    def __init__(self, A, B, *, A_integral_points=None, B_minus_A_integral_points=None, order=None):
        if isinstance(A, (tuple, list)):
            # Assume it's [halfA, A] where A is a Polyhedron
            self._A = A[1]
            self._halfA = A[0]
        else:
            self._A = A

        if A_integral_points is not None:
            self._A_integral_points = A_integral_points

        if B_minus_A_integral_points is not None:
            self._B_minus_A_integral_points = B_minus_A_integral_points
        elif isinstance(B, (tuple, list)):
            self._B_minus_A_integral_points = B
        else:
            self._B_minus_A_integral_points = tuple(v for v in B.integral_points() if v not in self._A_integral_points)

        self._order = order

    def __repr__(self):
        if self.gap():
            return f"Sandwich conv({sorted(self._A.vertices_list())}) with extra candidates {sorted(self._B_minus_A_integral_points)}) and gap {self.gap()}"
        return f"Polytope conv({sorted(self._A.vertices_list())})"

    def plot(self):
        return sum(point(p, size=30, color='yellow') for p in self._B_minus_A_integral_points) + self._A.plot(alpha=.3, polygon='red')

    @lazy_attribute
    def _halfA(self):
        return tuple(self._A.vertices_list())

    def B_minus_A_integral_points(self):
        return self._B_minus_A_integral_points

    @cached_method
    def gap(self):
        return len(self.B_minus_A_integral_points())

    def key_funcs(self):
        return (self._key_func_A_dimensions,
                self._key_func_A_partitions,
                self._key_func_A_palp_native_normal_form)

    @staticmethod
    def key_costs():
        return (0,
                1,
                50)

    @cached_method
    def noninvariant_keys(self):
        return (self._halfA,
                tuple(sorted(self._B_minus_A_integral_points)))


def dict_factory(key_prefix):
    return dict()


class SandwichStorage:
    r"""
    Minimal implementation of a dictionary with hierarchical lazy keys.

    Items must support the :class:`Sandwich_base` protocol.

    Strictly worse than a proper lazy trie because everything is stashed into large dictionaries.

    INPUT:

    - ``mapping_factory`` -- Constructor for a :class:`dict` or other mapping

    EXAMPLES::

        sage: d = SandwichStorage()
        sage: d['aaaa'] = 1
        sage: d._mapping_list
        sage: d['aaba'] = 2
        sage: d._mapping_list
        sage: d['aabb'] = 3
        sage: d._mapping_list
        sage: d['aaaa'] = 7  # FIXME: overwriting creates a long chain
        sage: d._mapping_list
    """
    def __init__(self, mapping_factory=None):
        if mapping_factory is None:
            mapping_factory = dict_factory
        self._mapping_factory = mapping_factory
        self._mapping_list = [mapping_factory(())]  # key_length -> key_prefix -> (key, value) | 'not_unique'

    def _key_prefix_to_mapping(self, key_prefix):
        length = len(key_prefix)
        while length >= len(self._mapping_list):
            self._mapping_list.append(self._mapping_factory(key_prefix[:len(self._mapping_list)]))
        return self._mapping_list[length]

    def _sufficient_key_prefix(self, key):
        r"""
        Return the shortest prefix of ``key`` that suffices to either:
        - identify a unique candidate for ``key``, in which case
          ``(key_prefix, (candidate_key, candidate_value), checked)`` is returned;
          when ``checked`` is True, the ``candidate_key`` is already a known hit.
        - show that ``key`` is not in ``self``, in which case ``(key_prefix, None, False)`` is returned

        OUTPUT: a tuple
        """
        key_prefix = ()
        while True:
            mapping = self._key_prefix_to_mapping(key_prefix)
            try:
                item = mapping[key_prefix]
            except KeyError:
                return key_prefix, None, False
            else:
                if item != 'not_unique':
                    return key_prefix, item, False

            # possible improvement: insisting on a unique candidate is too much when the next key element
            # is too expensive. When the subtrie has <= THRESHOLD candidates, it may be faster to invert:
            # loop through all candidates and do the fast non-invariant check.
            if (cost := self._key_cost(key, len(key_prefix))) > 1:
                next_mapping = self._key_prefix_to_mapping(key_prefix + (None,))
                if not isinstance(next_mapping, dict) or len(next_mapping) <= cost:  # len is expensive for diskcache.Index
                    for i, (same_prefix, item) in enumerate(next_mapping.items()):
                        if item != 'not_unique':
                            if item[0].__eq_noninvariant__(key):
                                return key_prefix + (self._key_item(item, 0),), item, True
                        if i >= cost:
                            break
            try:
                key_prefix = key_prefix + (self._key_item(key, len(key_prefix)),)
            except IndexError:
                assert False, 'path cannot end with not_unique'

    def _key_item(self, key, index):
        return key[index]

    def _key_cost(self, key, index):
        return key.item_cost(index)

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __getitem__(self, key):
        key_prefix, item, checked = self._sufficient_key_prefix(key)
        if item is None:
            raise KeyError(key)
        candidate_key, candidate_value = item
        if len(key_prefix) == len(key) or checked:
            return candidate_value
        if candidate_key.__eq_noninvariant__(key):
            return candidate_value
        for i in range(len(key_prefix), len(key)):
            if self._key_item(candidate_key, i) != self._key_item(key, i):
                raise KeyError(key)  # f'{key!r}; _sufficient_key_prefix returned {key_prefix=} {item=}')
        return candidate_value

    def __setitem__(self, key, value):
        key_prefix, item, checked = self._sufficient_key_prefix(key)
        mapping = self._key_prefix_to_mapping(key_prefix)
        if checked or item is not None:
            candidate_key, candidate_value = item
            while len(key_prefix) < len(key):
                candidate_next = candidate_key[len(key_prefix)]
                key_next = key[len(key_prefix)]

                candidate_key_prefix = key_prefix + (candidate_next,)
                candidate_mapping = self._key_prefix_to_mapping(candidate_key_prefix)
                candidate_mapping[candidate_key_prefix] = item

                mapping[key_prefix] = 'not_unique'  # atomic

                key_prefix = key_prefix + (key_next,)
                mapping = self._key_prefix_to_mapping(key_prefix)
                if candidate_next != key_next:
                    break
        mapping[key_prefix] = (key, value)

    def keys(self):
        for key, value in self.items():
            yield key

    __iter__ = keys

    def __len__(self):
        return len(list(iter(self)))  # FIXME obviously

    def values(self):
        for key, value in self.items():
            yield value

    def items(self):
        key_prefix = []  # currently only the length matters
        not_unique = True
        while not_unique:
            not_unique = False
            mapping = self._key_prefix_to_mapping(key_prefix)
            for item in mapping.values():
                if item == 'not_unique':
                    not_unique = True
                else:
                    yield item
            if not_unique:
                key_prefix.append(None)


def break_symmetry(A,m):
    """
       takes a centrally symmetric m-dimensional polytope A
       computes a subset halfA of its vertices I such that I = conv(halfA \cup -halfA)
    """
    halfA = []
    for z in A.vertices():
        if next((x for x in z if x != 0), None) > 0:
            l = tuple(int(x) for x in z)
            halfA.append(l)
    return tuple(sorted(halfA))


def do_not_break_symmetry(A, m):
    return tuple(sorted(tuple(int(x) for x in z) for z in A.vertices()))


def is_extendable(S,v,Delta):
    """
        Check whether the extension of a set S of vectors by a vector v causes a determinant to exceed Delta.
    """
    m = len(v)
    for C in Combinations(S,m-1):
        M = matrix(C + [list(v)])
        if abs(det(M)) > Delta:
            return False
    return True


def is_extendable_pair(S,v,w,Delta):
    """
        Check whether the extension of a set S of vectors by a vector v causes a determinant to exceed Delta.
    """
    m = len(v)
    M = matrix(S)
    for C in Combinations(S, m-2):
        M = matrix(C + [list(v)] + [list(w)])
        if abs(det(M)) > Delta:
            return False
    return True


def layered_polytope_from_sandwich(A,B):
    """ 3*B is embedded into height 0, two copies of 3*A are embedded into heights 1 and -1.
        Then, one generates a polytope based on these three layers at heights -1,0 and 1
        Note: If A and B are centrally symmetric, then the resulting polytope is centrally symmetric as well.
    """
    middleLayer = [tuple(3*vector(v))+(0,) for v in B.vertices()]
    upperLayer = [tuple(3*vector(v))+(1,) for v in A[1].vertices()]
    lowerLayer = [tuple(3*vector(v))+(-1,) for v in A[1].vertices()]
    return Polyhedron(middleLayer+upperLayer+lowerLayer, backend=B.backend())


# Sandwich factory is used to store sandwiches up to affine unimodular transformations.
# A sandwich factory is a dictionary of dictionaries. For each possible gap, a storage
# for sandwiches with this gap is created. The latter storage
# is a dictionary with key,value pairs such that the value is a sandwich and
# the respective key is the sandwich normal form of this sandwich.


class SandwichFactory_base:

    @abstract_method
    def prepare_sandwich(self):
        pass

    @abstract_method
    def reduce_sandwich(self, sandwich, *args, **kwds):
        pass

    def append_sandwich(self, sandwich):
        """
            If no affine unimodular image of the sandwich (A,B) is in the sandwich factory self,
            the sandwich (A,B) is appended to self.
        """
        Gap = sandwich.gap()

        # crucial that sandwich is a LatticePolytope (or something else with a good hash),
        # not a Polyhedron (which has a poor hash)
        if sandwich not in self[Gap]:
            self[Gap][sandwich] = (sandwich._halfA, sandwich._A)
            # self[Gap][sandwich] = [(sandwich._halfA, sandwich._A), sandwich._B]
            if not Gap:
                print(sandwich)
            self.sandwich_failures += 1
            return sandwich
        else:
            self.sandwich_hits += 1
            return None

    def __repr__(self):
        return f'{self.__class__.__name__} with keys {sorted(self)}'

    @abstract_method
    def branch_sandwich(self, sandwich, *args, **kwds):
        pass

    def sandwich_factory_statistics(self):
        logging.info("Maximum gap in sandwiches: %d",max(self.keys()))
        logging.info("Number of sandwiches: %d",sum([len(self[Gap]) for Gap in self.keys() if Gap!=0]))
        if 0 in self.keys():
            logging.info("Number of polytopes found: %d", len(self[0]))
        logging.info(f"Sandwich normal form hits: {self.sandwich_hits}, failures: {self.sandwich_failures}")
        logging.info(50*"-")


class SandwichFactory(defaultdict, SandwichFactory_base):

    def __init__(self, m, Delta, mode, polyhedra_backend='ppl'):
        super().__init__(SandwichStorage)
        self._m = m
        self._Delta = Delta

        # Normalize computation mode
        if not mode:
            mode = 'delta'
        elif mode is True:
            mode = 'delta_ext'

        # if mode not in ['delta', 'delta_ext', 'delta_cone']:
        #     raise ValueError("Unknown computation mode", mode)

        self._mode = mode
        if mode == 'delta_ext':
            # set the known lower bound for h(Delta,m) by Lee et al.
            self._cmax = m^2 - m + 1 *2*m*Delta
        self._deque = []

        self._polyhedra_backend = polyhedra_backend
        self._polyhedra_parent = Polyhedra(ZZ, m, backend=polyhedra_backend)
        self.sandwich_hits = 0
        self.sandwich_failures = 0

    def prepare_sandwiches(self):
        m = self._m
        Delta = self._Delta
        mode = self._mode

        if Delta == 2:
            HNFs = []
            for nonzeros in range(m):
                R = matrix.identity(m)
                for i in range(m-nonzeros-1,m):
                    R[i, m-1] += 1
                HNFs.append(R)
        else:
            HNFs = delta_normal_forms(m,Delta)

        for basisA in HNFs:
            # first, we generate A and halfA out of basisA
            mbA = matrix(basisA)
            mA = mbA.augment(-mbA)
            A_points = mA
            A = self._polyhedra_parent([A_points.transpose(), [], []], None, convert=True)

            halfA = break_symmetry(A,m)

            # second, the outer container B is the centrally symmetric parallelotope spanned by the vectors in basisA
            B = polytopes.parallelotope(mA.transpose(), backend=self._polyhedra_backend)

            # B may contain some integral points that are Delta-too-large with respect to A, and so we do:
            sandwich = Sandwich([halfA,A], B)
            yield self.reduce_sandwich([halfA,A], sandwich)

    def reduce_sandwich(self, newA, sandwich):
        """
        For a given sandwich (A,B) and a value of Delta
        the function returns a polytope
        obtained by removing all of the lattice points v of B
        with the property that if v is added to A, there will be a determinant of absolute value > Delta
        """
        Delta = self._Delta

        to_be_removed = set()
        to_be_kept = set()

        Z = sandwich.B_integral_points()
        for v in Z:
            if v in to_be_removed or v in to_be_kept:  ## this just avoids considering -w in case that w was considered already before
                continue
            if v in newA[1]:
                continue
            mv = -v
            mv.set_immutable()
            if is_extendable(newA[0],v,Delta):
                to_be_kept.add(v)
                to_be_kept.add(mv)
            else:
                to_be_removed.add(v)
                to_be_removed.add(mv)
        if to_be_removed:
            newB = tuple(z for z in Z if z not in to_be_removed)
            return Sandwich(newA, newB)
        else:
            return Sandwich(newA, sandwich._B, B_integral_points=Z)

    def append_sandwich(self, sandwich):
        """
            If no affine unimodular image of the sandwich (A,B) is in the sandwich factory self,
            the sandwich (A,B) is appended to self.
        """
        Gap = sandwich.gap()

        # crucial that sandwich is a LatticePolytope (or something else with a good hash),
        # not a Polyhedron (which has a poor hash)
        if sandwich not in self[Gap]:
            self[Gap][sandwich] = [(sandwich._halfA, sandwich._A), sandwich._B]
            if not Gap:
                print(sandwich)
            self.sandwich_failures += 1
            return sandwich
        else:
            self.sandwich_hits += 1
            return None

    def branch_sandwich(self, sandwich, B_v_order=None):

        A = (sandwich._halfA, sandwich._A)
        B = sandwich._B

        for v in B.vertices(): # pick a vertex of B which is not in A
            if v not in A[1]:
                break

        v = vector(v, immutable=True)
        mv = -v
        points_added = [v, mv]

        blow_up_of_A = self._polyhedra_parent([list(A[1].vertices()) + points_added, [], []],
                                            None,
                                            convert=True)  ## this uses that all points in B are "Delta-ok" for A
        half_of_blow_up_of_A = break_symmetry(blow_up_of_A, self._m)
        newA = [half_of_blow_up_of_A, blow_up_of_A]
        sandwich1 = self.reduce_sandwich(newA, sandwich)

        reduction_of_B = tuple(z for z in sandwich.B_integral_points()
                            if z not in points_added)
        sandwich2 = Sandwich(A, reduction_of_B, A_integral_points=sandwich.A_integral_points())
        if self._mode == 'delta_ext':
            if sandwich1.B_integral_points_count() >= self._cmax:
                yield sandwich1
                npts_blow_up = sandwich1.A_integral_points_count()
                if npts_blow_up > self._cmax:
                    self._cmax = npts_blow_up
            if sandwich2.B_integral_points_count() >= self._cmax:
                yield sandwich2
        else:
            yield sandwich1
            yield sandwich2


class SandwichFactory_with_order(defaultdict, SandwichFactory_base):

    def __init__(self, m, Delta, polyhedra_backend='ppl', B_v_order=None):
        super().__init__(SandwichStorage)
        self._m = m
        self._Delta = Delta

        # Normalize computation mode
        # if not mode:
        #     mode = 'delta'
        # elif mode is True:
        #     mode = 'delta_ext'

        # if mode not in ['delta', 'delta_ext', 'delta_cone']:
        #     raise ValueError("Unknown computation mode", mode)

        # self._mode = mode
        # if mode == 'delta_ext':
        #     # set the known lower bound for h(Delta,m) by Lee et al.
        #     self._cmax = m^2 - m + 1 *2*m*Delta
        self._deque = []

        self._polyhedra_backend = polyhedra_backend
        self._polyhedra_parent = Polyhedra(ZZ, m, backend=polyhedra_backend)
        self.sandwich_hits = 0
        self.sandwich_failures = 0

        self._B_all = tuple()

        if not B_v_order:
            B_v_order = 'full'
        if B_v_order not in ['lex', 'lex_reverse', 'large_norm_first', 'small_norm_first', 'full']:
            raise ValueError("Unknown computation mode", B_v_order)
        self._B_v_order = B_v_order

    def prepare_sandwiches(self):
        m = self._m
        Delta = self._Delta
        # mode = self._mode

        if Delta == 2:
            HNFs = []
            for nonzeros in range(m):
                R = matrix.identity(m)
                for i in range(m-nonzeros-1,m):
                    R[i, m-1] += 1
                HNFs.append(R)
        else:
            HNFs = delta_normal_forms(m,Delta)

        for basisA in HNFs:
            # first, we generate A and halfA out of basisA
            mbA = matrix(basisA)
            # Start with positive HNF vectors only; this breaks symmetry
            A_points = mbA.augment(vector(ZZ, m))
            A = self._polyhedra_parent([A_points.transpose(), [], []], None, convert=True)
            halfA = A_points.columns()

            mA = mbA.augment(-mbA)

            # second, the outer container B is the centrally symmetric parallelotope spanned by the vectors in basisA
            B = polytopes.parallelotope(mA.transpose(), backend=self._polyhedra_backend)

            # B may contain some integral points that are Delta-too-large with respect to A, and so we do:
            sandwich = Sandwich_with_order([halfA,A], B, order=-1)
            sandwich_new = self.reduce_sandwich([halfA,A], sandwich, order=-1)
            self._B_all += tuple(v for v in sandwich_new.B_minus_A_integral_points() if v not in self._B_all)
            yield sandwich_new

    @lazy_attribute
    def _B_v_order_dict(self):
        B_v_order_dict = {}
        match self._B_v_order:
            case 'lex':
                for i, v in enumerate(sorted(self._B_all)):
                    B_v_order_dict[v] = i
            case 'lex_reverse':
                for i, v in enumerate(sorted(self._B_all, reverse=True)):
                    B_v_order_dict[v] = i
            case 'small_norm_first':
                for i, v in enumerate(sorted(self._B_all, key=lambda x: x.norm(1))):
                    B_v_order_dict[v] = i
            case 'large_norm_first':
                for i, v in enumerate(sorted(self._B_all, key=lambda x: x.norm(1), reverse=True)):
                    B_v_order_dict[v] = i
            case 'full':
                for v in self._B_all:
                    B_v_order_dict[v] = -1

        for v in B_v_order_dict:
            mv = -v
            mv.set_immutable()
            if mv in B_v_order_dict:
                B_v_order_dict[mv] = B_v_order_dict[v]
        return B_v_order_dict

    def B_v_order_dict(self):
        return self._B_v_order_dict

    def reduce_sandwich(self, newA, sandwich, order=None):
        """
        For a given sandwich (A,B) and a value of Delta
        the function returns a polytope
        obtained by removing all of the lattice points v of B
        with the property that if v is added to A, there will be a determinant of absolute value > Delta
        """
        Delta = self._Delta
        # mode = self._mode

        to_be_removed = set()

        Z = sandwich.B_minus_A_integral_points()
        for v in Z:
            if v in to_be_removed:  ## this just avoids considering -w in case that w was considered already before
                continue
            if v in newA[1]:
                to_be_removed.add(v)
                continue
            mv = -v
            mv.set_immutable()
            if mv/1000 in newA[1]:
                # never extend to a non-pointed cone
                to_be_removed.add(v)
                continue
            if not is_extendable(newA[0],v,Delta):
                to_be_removed.add(v)
                to_be_removed.add(mv)
        if to_be_removed:
            newB = tuple(z for z in Z if z not in to_be_removed)
            return Sandwich_with_order(newA, newB, order=order)
        else:
            return Sandwich_with_order(newA, Z, order=order)

    def branch_sandwich(self, sandwich):

        B_v_order = self.B_v_order_dict()
        A = (sandwich._halfA, sandwich._A)

        for v in sorted(sandwich.B_minus_A_integral_points(), key=lambda x: B_v_order[x]): # pick any integral point in B which is not in A with any given ordering B_v_order
            if sandwich._order > -1 and B_v_order[v] <= sandwich._order:
                continue
            blow_up_of_A = self._polyhedra_parent([list(A[1].vertices()) + [vector(v, immutable=True)], [], []],
                                        None,
                                        convert=True)
            half_of_blow_up_of_A = do_not_break_symmetry(blow_up_of_A, self._m)
            newA = [half_of_blow_up_of_A, blow_up_of_A]
            sandwich_i = self.reduce_sandwich(newA, sandwich, order=B_v_order[v])
            if all(B_v_order[vv] <= sandwich_i._order for vv in sandwich_i.B_minus_A_integral_points()):
                if any(B_v_order[vv] < sandwich_i._order for vv in sandwich_i.B_minus_A_integral_points()):
                    continue
            yield sandwich_i


def new_sandwich_factory(m, Delta, mode, dirname=None, B_v_order=None, **kwds):

    # Using https://github.com/mina86/pygtrie (https://pygtrie.readthedocs.io/en/latest/#pygtrie.Trie)
    # seemed promising, but unfortunately it always eagerly uses the whole key
    # when creating a new node (in _set_node).
    # (Our SandwichStorage does that only when we overwrite an item, which
    # we never do here.)
    #from pygtrie import Trie
    #sandwich_factory = defaultdict(Trie)

    if dirname is None:
        match mode:
            case 'delta_cone':
                sandwich_factory = SandwichFactory_with_order(m, Delta, B_v_order=B_v_order, **kwds)
            case _:
                sandwich_factory = SandwichFactory(m, Delta, mode, **kwds)

    return sandwich_factory


def delta_classification(m, Delta, mode, B_v_order=None, dirname=None, *, order='gap', iterations=None,
                         polyhedra_backend='ppl'):
    """
    Run the sandwich factory algorithm.

    INPUT:

    - ``mode`` -- one of

      - ``'delta'`` -- classify all centrally symmetric m-dimensional lattice polytopes
        with largest determinant equal to Delta

      - ``'delta_ext'`` -- only include the extremal examples attaining h(Delta,m)

      - ``'delta_cone' -- oriented, non--centrally symmetric version (`conv(A\cup\{0\})`
        such that `\{x: Ax=0, x\ge 0\}=\{0\}`)
    """
    sf = new_sandwich_factory(m, Delta, mode, B_v_order=B_v_order, dirname=dirname,
                              polyhedra_backend=polyhedra_backend)

    match order:
        case 'gap':
            for sandwich in sf.prepare_sandwiches():
                sf.append_sandwich(sandwich)

            maxGap = max(sf.keys())
            while maxGap > 0:
                sf.sandwich_factory_statistics()
                for sandwich in sf[maxGap]:
                    for new_sandwich in sf.branch_sandwich(sandwich):
                        sf.append_sandwich(new_sandwich)
                del sf[maxGap]
                maxGap = max(sf.keys())
            sf.sandwich_factory_statistics()

        case _:
            deque = sf._deque
            for sandwich in sf.prepare_sandwiches():
                if sf.append_sandwich(sandwich) is not None:
                    deque.append(sandwich)

            iteration = 0

            while deque:
                iteration += 1
                match order:
                    case 'dfs':
                        sandwich = deque.pop()
                    case 'bfs':
                        sandwich = deque.popleft()
                    case 'random':
                        with deque.transact():
                            index = randint(0, len(deque) - 1)
                            sandwich = deque[index]
                            deque[index] = None
                    case _:
                        raise ValueError(f'unknown order parameter: {order}')
                if sandwich is None:
                    continue
                for new_sandwich in sf.branch_sandwich(sandwich):
                    if sf.append_sandwich(new_sandwich) is not None:
                        if new_sandwich.gap():
                            deque.append(new_sandwich)

                if iteration % 2000 == 0:
                    sf.sandwich_factory_statistics()  # very expensive when using diskcache.Deque

                if iterations is not None and iteration > iterations:
                    break

    result = []
    for A in sf[0].values():
        result.append(A[1])  ## only store the polytope in A

    return result


def plot_delta_classification(m, Delta=None, mode=None, L=None):
    match m:
        case 2r:
            return graphics_array([P.plot(xmin=-Delta, xmax=Delta,
                                          ymin=-Delta, ymax=Delta,
                                          axes=True, ticks=[[], []],
                                          gridlines=[range(-Delta,Delta+1),
                                                     range(-Delta,Delta+1)])
                                   for P in L],
                                  ncols=6)
        case 3r:
            G = Graphics()
            L_iter = iter(L)
            V = RDF^3
            try:
                for y in range(isqrt(len(L))):
                    for x in range(isqrt(len(L)) + 1):
                        P = next(L_iter)
                        center = V([x * (2*Delta+1), y * (2*Delta+1), 0])
                        # coordinate planes
                        for i, j in Combinations(3, 2):
                            G += polygon([center - Delta*V.gen(i) - Delta*V.gen(j),
                                          center - Delta*V.gen(i) + Delta*V.gen(j),
                                          center + Delta*V.gen(i) + Delta*V.gen(j),
                                          center + Delta*V.gen(i) - Delta*V.gen(j)],
                                         color='grey', alpha=.1)
                        P_shifted = P + center
                        G += P_shifted.plot(xmin=-Delta, xmax=Delta,
                                            ymin=-Delta, ymax=Delta,
                                            zmin=-Delta, zmax=Delta,
                                            axes=False, alpha=.3, polygon='red')
            except StopIteration:
                pass
            return G


def delta_submatrix(S, m, Delta):
    for C in Combinations(S, m):
        HNF = matrix(C)
        if abs(det(HNF)) == Delta:
            return HNF.transpose()
    return None


def is_maximal(A, m, Delta, HNF=None, certificate=False):
    P_all = A.integral_points()
    halfA = break_symmetry(A, m)
    if not HNF:
        HNF = delta_submatrix(halfA, m, Delta)
        if not HNF:
            raise ValueError
    mA = HNF.augment(-HNF)
    B = polytopes.parallelotope(mA.transpose())
    for v in B.integral_points():
        if not v in P_all:
            if is_extendable(halfA, v, Delta):
                if certificate:
                    return False, v
                else:
                    return False
    if certificate:
        return True, A
    else:
        return True


def update_delta_classification_database(m,Delta,mode):
    # the files storing polytopes are created in the data subfolder
    if not os.path.exists('data'):
        os.mkdir('data')

    # let's see whether the file for the pair (m,Delta) is missing
    match mode:
        case 'delta':
            missingDelta = not os.path.isfile(FILE_NAME_DELTA % (m,Delta))
        case 'delta_ext':
            missingDelta = not os.path.isfile(FILE_NAME_DELTA_EXTR % (m,Delta))
        case 'delta_max':
            missingDelta = not os.path.isfile(FILE_NAME_DELTA_MAX % (m,Delta))
        case 'delta_cone':
            missingDelta = not os.path.isfile(FILE_NAME_DELTA_CONE % (m,Delta))

    if missingDelta:
        # we should run the delta classification

        match mode:
            case 'delta':
                result = delta_classification(m,Delta,mode)
                f = open(FILE_NAME_DELTA % (m,Delta),'w')
                print([[tuple(p) for p in P.vertices()] for P in result],file=f)
                f.close()
            case 'delta_max':
                if not (os.path.isfile(FILE_NAME_DELTA % (m,Delta))):
                    result = delta_classification(m,Delta,'delta')
                    f = open(FILE_NAME_DELTA % (m,Delta),'w')
                    print([[tuple(p) for p in P.vertices()] for P in result],file=f)
                    f.close()
                f = open(FILE_NAME_DELTA_MAX % (m,Delta),'w')
                g = open(FILE_NAME_DELTA % (m,Delta),'r')
                L = eval(g.read().replace('\n',' '))
                g.close()
                hdm = generalized_heller_constant(m,Delta,false)[0]
                result = []
                for P in L:
                    if is_maximal(Polyhedron(P), m, Delta):
                        result.append(P)
                print([P for P in result],file=f)
                f.close()

            case 'delta_ext':
                f = open(FILE_NAME_DELTA_EXTR % (m,Delta),'w')
                if (os.path.isfile(FILE_NAME_DELTA % (m,Delta))):
                    g = open(FILE_NAME_DELTA % (m,Delta),'r')
                    L = eval(g.read().replace('\n',' '))
                    g.close()
                    hdm = generalized_heller_constant(m,Delta,false)[0]
                    result = []
                    for P in L:
                        if (Polyhedron(P).integral_points_count() == hdm):
                            result.append(P)
                    print([P for P in result],file=f)
                    f.close()
                else:
                    result = delta_classification(m,Delta,extremal)
                    print([[tuple(p) for p in P.vertices()] for P in result],file=f)
                    f.close()
            case 'delta_cone':
                result = delta_classification(m,Delta,mode)
                f = open(FILE_NAME_DELTA_CONE % (m,Delta),'w')
                print([[tuple(p) for p in P.vertices()] for P in result],file=f)
                f.close()


def lattice_polytopes_with_given_dimension_and_delta(m,Delta,mode):
    """
    That's the main function for users of this module. It returns the list of all [extremal=false] or only h(Delta,m)-attaining [extremal=true]
    m-dimensional centrally symmetric lattice polytopes with delta equal to Delta.

    INPUT:

    - ``mode`` -- one of

      - ``'delta'`` or ``False`` -- classify all centrally symmetric m-dimensional lattice polytopes
        with largest determinant equal to Delta

      - ``'delta_max'`` or ``True`` -- only include the inclusion maximal examples

      - ``'delta_ext'`` -- only include the extremal examples attaining h(Delta,m)

      - ``'delta_cone' -- oriented, non--centrally symmetric version (`conv(A\cup\{0\})`
        such that `\{x: Ax=0, x\ge 0\}=\{0\}`)
    """
    if not mode:
        mode = 'delta'
    elif mode is True:
        mode = 'delta_ext'

    if mode not in ['delta', 'delta_ext', 'delta_max', 'delta_cone']:
        raise ValueError("Unknown computation mode", mode)
    # first, we update the database of lattice polytopes with a given delta
    update_delta_classification_database(m,Delta,mode)

    # now, we can read the list of polytopes from the corresponding file and return them
    match mode:
        case 'delta':
            f = open(FILE_NAME_DELTA % (m,Delta),'r')
        case 'delta_ext':
            f = open(FILE_NAME_DELTA_EXTR % (m,Delta),'r')
        case 'delta_max':
            f = open(FILE_NAME_DELTA_MAX % (m,Delta),'r')
        case 'delta_cone':
            f = open(FILE_NAME_DELTA_CONE % (m,Delta),'r')

    L = eval(f.read().replace('\n',' '))
    f.close()
    return [Polyhedron(P) for P in L]


## Code below uses boolean "extremal"; above has been generalized to "mode"


def generalized_heller_constant(m,Delta,extremal):
    """
        Compute the generalized Heller constant h(Delta,m) and a point set attaining it
    """

    if not extremal:
        mode = 'delta'
    else:
        mode = 'delta_ext'

    DeltaPolytopes = lattice_polytopes_with_given_dimension_and_delta(m,Delta,mode)
    nmax = 0
    for P in DeltaPolytopes:
        npoints = P.integral_points_count()
        if npoints > nmax:
            nmax = npoints
            Pmax = P
    return nmax , Pmax, len(DeltaPolytopes)
