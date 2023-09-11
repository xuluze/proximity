#####################################################################################################
# this is https://github.com/christopherborger/mixed_volume_classification/blob/master/polytopes.sage
#####################################################################################################

from collections import defaultdict

# ========================================================================
# Functions on polytopes which are not directly
# available as methods in the Polyhedron class
# ========================================================================


def is_difference_set(P,summand=true):
    """
        Checks wether the polytope P can be written as P = Q - Q for another polytope Q.
        The boolean argument summand determines if Q in positive case is returned as well.
    """
    for A,B in P.minkowski_decompositions():
        if (B - B.centroid() == -(A - A.centroid())):
            if summand:
                return true , B
            else:
                return true
    if summand:
        return false , Polyhedron([])
    else:
        return false


def embed_polygon(P):
    """
        Embed a polygon into height 0 in dimension three.
        NOTE: this can be used for polytopes, too, but we use it only for polygons.
    """
    return Polyhedron([list(v) + [0] for v in P.vertices()])

def affine_hull(A):
    """
        affine hull of a polyhedron, represented as a Polyhedron object.
    """
    return Polyhedron(ieqs=[vector(e) for e in A.equations()]+[-vector(e) for e in A.equations()])


def ineq_face(P,ineq):
    V=[v for v in P.vertices() if ineq.eval(v)==0]
    return Polyhedron(vertices=V)

def support_function(P,u):
    u=vector(u)
    return max([u.dot_product(v) for v in map(vector,P.vertices())])

def facets_and_outer_normals(P):
    """
        generator of pairs facet,outer_normal_of_facet
    """
    for ineq in P.Hrepresentation():
        F=ineq_face(P,ineq)
        normalF=-vector(ineq[1:])
        yield F,normalF

def interior_integral_points(P):
    return [z for z in P.integral_points() if P.interior_contains(z)]

def as_full_dim_polyhedron(P):
    P = P.lattice_polytope()

    L = P.lattice()
    V = P.vertices()

    if L.zero() not in V:
        V = [v-V[0] for v in V]

    S = L.span(V).saturation()

    return Polyhedron([S.coordinates(v) for v in V])


def Volume(P):
    """
        normalized volume (in the sense of the lattice polytope theory)
    """
    return factorial(P.dimension())*P.volume()

def relVolume(P):
    """
        relative normalized volume
    """
    return Volume(as_full_dim_polyhedron(P))


def affine_normal_form(P):
    V=[vector(v) for v in P.vertices()]
    s=sum(V)
    N=len(V)
    Q=N*P-s
    try:
        # in case PALP (hidden behind normal_form()) has some trouble to handle Q, we'll catch the exception and see what P was
        Q=Polyhedron(Q.lattice_polytope().normal_form())
    except Exception as e:
        print("FINALLY CAUGHT ONE!!")
        Q = Polyhedron(Q.lattice_polytope().normal_form(algorithm='palp_native'))
#        print("ERROR MESSAGE:")
#        print("affine_normal_form(P) failed on the polytope P with the vertices")
#        print(map(tuple,P.vertices()))
#        print("Here is the exception:")
#        print(e)
#        print("END OF THE ERROR MESSAGE")
    Q=Q-vector(min(Q.vertices()))
    return Q/N


def translative_normal_form(P):
    """
        Return P minus the lexicographically minimal vertex of P.
        Properties of the returned polytope:
            if P and Q coincide up to translations, then their translative normal form is the same.
            the zero is always a vertex of the returned polytope.

    """
    return P-vector(min(P.vertices()))


# =============================================
# Generating special polytopes
# =============================================

def std_simplex(d):
    return Polyhedron([d*(0,)]+identity_matrix(d).rows())


def delta_crosspolytopes(m,Delta):
    """
        returns a list of bases of all m-dimensional crosspolytopes with determinant equal to Delta
    """
    # TODO: this function could have been implemented as a generator.
    FILE_NAME='data/hnfs-%d-%d.sage' % (m,Delta)
    load(FILE_NAME)

    return HNFs


def delta_normal_forms(m,Delta):
    """
    	returns a list of all mxm matrices in Hermite Normal Form and with determinant equal to Delta
    """
    if m == 1:
        return [ [[Delta]] ]
    else:
        HNFs = []
        for d in divisors(Delta):
            lowdimHNFs = delta_normal_forms(m-1,Integer(Delta/d))
            for H in lowdimHNFs:
                ROWS = [[d]]	# generate new rows and only afterwards generate new matrices
                for i in range(m-1):
                    ROWS = [ r + [j] for r in ROWS for j in range(H[i][i]) ]
                body = [[0] + x for x in H]
                for R in ROWS:
                    HNFs.append([R] + body)
        return HNFs


# ==========================================
# Input/Output for polytopes
# ==========================================

def save_polytopes(polytopes,fname):
    f=open(fname,'w')
    print >>f, [map(tuple,P.vertices()) for P in polytopes]
    f.close()

def polytopes_from_file(fname):
    f=open(fname,'r')
    L=eval(f.read().replace('\n',' '))
    L=list(map(Polyhedron,L))
    f.close()
    return L









