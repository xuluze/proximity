load("matrix-extraction.sage")
from itertools import product

def ones_vector(size):
    """
    Returns a vector in RR^size with all entries equal to 1.
    """
    return vector([1 for i in range(size)])

def orthogonal_decomp_given_matrix(v,M):
    """
    Computes the orthogonal decomposition of v = u + w,
    where u is in RowSpace(M) and w is in Kernel(M^T).
    """
    proj_M = M*(M.transpose()*M).inverse()*M.transpose()
    u = proj_M*v
    w = v - u
    return u,w

def Extend_Polytope_To_Full_Dimension(P):
    """
    This function takes a polytope and adds width in the appropriate
    dimensions such that the resulting polytope is full dimensional.
    If a full dimensional polytope is entered, it is returned.
    """
    L = [(eqns.b() + 1,) + tuple(eqns.A()) for eqns in P.equations()] + [(-eqns.b() + 1,) + tuple(-eqns.A()) for eqns in P.equations()]

    M = []
    A_eq = matrix(tuple(eqns.A()) for eqns in P.equations())
    b = vector(eqns.b() for eqns in P.equations())
    for ieqs in P.inequalities():
        d = ieqs.b()
        a_T = ieqs.A()
        u,w = orthogonal_decomp_given_matrix(a_T,A_eq.transpose())
        y = A_eq.solve_left(u)
        M.append((d - y*b,) + tuple(w))

    P_new = Polyhedron(ieqs = L + M, backend = 'normaliz')
    return P_new

def Delta_modularity(A):
    if A.ncols() <= A.nrows():
        A = A.transpose()

    return max(abs(x) for x in A.minors(A.nrows()))

def Basis_Extraction_Given_Matrix(A):
    """
    Given a matrix A, this function determines all of the maximal-sized invertible
    square submatrices of A and returns them in a list.
    """
    from sage.matroids.advanced import BasisMatroid

    if A.ncols() >= A.nrows():
        A_mat = Matroid(matrix = A)
        return BasisMatroid(A_mat).bases()
    else:
        A_mat = Matroid(matrix = A.transpose())
        return BasisMatroid(A_mat).bases()

def Unit_Lattice(A, exclude_zero=True):
    r"""
    This function returns, in a list, all of the vectors in the
    lattice A*[0,1)^n. (A is n by n.)

    EXAMPLES::

        sage: Unit_Lattice(matrix(ZZ, [[1, 0, 1], [0, 1, 1], [0, 0, 3]]))
        [(1, 1, 1), (1, 1, 2)]

    """
    from itertools import product

    D, U, V = A.smith_form() # UAV = D, A = U^-1 D V^-1, A^-1 = V D^-1 U

    D_lattice = tuple(list(range(d)) for d in D.diagonal())

    UL = []
    for i in product(*D_lattice):
        d = vector(i)
        if exclude_zero and d.norm(1) == 0:
            continue
        gi = V*D.inverse()*d
        UL.append(A*vector(x - x.floor() for x in gi))

    return(UL)

def proximity_norm(z_N, A_B_inv_A_N):
#     return z_N.norm(1) + (A_B_inv_A_N*z_N).norm(1)
    return max([z_N.norm(Infinity), (A_B_inv_A_N*z_N).norm(Infinity)])

def Polytope_given_B1_b(A_B_inv_A_N, s, A_B_inv_b, bound_N=None):
    r"""
    This function creates the polytope P(A_B_inv_A_N, s, A_B_inv_b) corr. to
    I_m * x_B + A_B_inv_A_N x_N = A_B_inv_b, x_N >= 0, x_{B(s)} >=0, with potentially x_N <= bound_N

    Return the same polyhedron as Polytope_given_B1_b_A

    EXAMPLES::

        sage: A = matrix(ZZ,[[0,1,1,1,2,2,2,3],[1,0,1,2,2,3,4,4]])
        sage: B = [5,6]
        sage: b = vector((1,2))
        sage: P1 = Polytope_given_B1_b_A(A, B, [], b, bound_N=vector([3,3,3,3,3,3]))
        sage: A_B = A.matrix_from_columns(B)
        sage: A_N = A.matrix_from_columns(i for i in range(A.ncols()) if i not in B)
        sage: P2 = Polytope_given_B1_b(A_B.inverse()*A_N, [], A_B.inverse()*b, bound_N=vector([3,3,3,3,3,3]))
        sage: P1 == P2
        True

    """
    m = A_B_inv_A_N.nrows()
    N_size = A_B_inv_A_N.ncols() # N_size = n - m
    n = m + N_size

    Ineqs = [(0,) + tuple(1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
    if bound_N:
        if isinstance(bound_N, sage.modules.vector_integer_dense.Vector_integer_dense):
            Ineqs = Ineqs + [(bound_N[i],) + tuple(-1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
        else:
            Ineqs = Ineqs + [(bound_N,) + tuple(-1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
    if s:
        Ineqs = [(0,) + tuple(1 if j == i else 0 for j in range(n)) for i in s] + Ineqs

    Equals = [(-A_B_inv_b[i],) + tuple(1 if j == i else 0 for j in range(m)) + tuple(A_B_inv_A_N[i]) for i in range(m)]
    return Polyhedron(ieqs = Ineqs, eqns = Equals, base_ring=QQ, backend='normaliz')

def Polytope_given_B1_b_A(A, B, s, b, bound_N=None):
    """
    This function creates the polytope P(A_B_inv_A_N, s, A_B_inv_b) corr. to
    A_B * x_B + A_N x_N = b, x_N >= 0, x_{B(s)} >=0, with potentially x_N <= bound_N
    """
    m = A.nrows()
    n = A.ncols()
    N_size = n - m

    A_B = A.matrix_from_columns(B)
    A_N = A.matrix_from_columns(frozenset(range(n)).difference(B))

    Ineqs = [(0,) + tuple(1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
    if bound_N:
        if isinstance(bound_N, sage.modules.vector_integer_dense.Vector_integer_dense):
            Ineqs = Ineqs + [(bound_N[i],) + tuple(-1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
        else:
            Ineqs = Ineqs + [(bound_N,) + tuple(-1 if j == m + i else 0 for j in range(n)) for i in range(N_size)]
    if s:
        Ineqs = [(0,) + tuple(1 if j == i else 0 for j in range(n)) for i in s] + Ineqs

    Equals = [(-b[i],) + tuple(A_B[i]) + tuple(A_N[i]) for i in range(m)]
    return Polyhedron(ieqs = Ineqs, eqns = Equals, base_ring=QQ, backend='normaliz')

def Polyhedral_Cone_A(A):
    """
    This function creates the polytope Ax = 0, x>=0
    """
    m = A.nrows()
    n = A.ncols()

    Equals = [(0,) + tuple(A[i]) for i in range(m)]
    Ineqs = [(0,) + tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]

    return Polyhedron(ieqs = Ineqs, eqns = Equals)

def is_dominated(v, w):
    """Check if v dominates w, i.e., v_i >= w_i for all i."""
    return all(v[i] >= w[i] for i in range(len(v)))

def minimal_vectors(vector_set):
    """Find all minimal vectors in the set."""
    minimal_set = []
    for v in vector_set:
        if not any(is_dominated(v, w) for w in vector_set if v != w):
            minimal_set.append(v)
    return minimal_set

def Zero_Step_b_hull(A_B_inv_A_N, b2, bound_N=None, verbose=False, check_obj_c=False):
    m = len(b2)
    n = A_B_inv_A_N.ncols() + m
    P = Polytope_given_B1_b(A_B_inv_A_N, [], b2)
    PI = P.integral_hull()

    if PI.is_empty() == True:
        return []  

    if check_obj_c:
        nonpos_orthant = Cone(-matrix.identity(n - m))
        PI_N = Polyhedron(vertices=[v[m:n] for v in PI.vertices()], base_ring=QQ, backend='normaliz')
        if PI_N.dimension() == 0:
            if verbose:
                print('Dimension 0. Single Vertex Case.')
            return [v.vector() for v in PI_N.vertices()]
        if PI_N.is_full_dimensional() == False and PI_N.dimension() > 0:
            if verbose:
                print('Zero Step Extend needed')
            P_new = Extend_Polytope_To_Full_Dimension(PI_N)
        else:
            if verbose:
                print('Zero Step Extend unnecessary')
            P_new = PI_N
    
        # If there exists a nonpos_c_N in the cone containing a vertex, return that vertex
        nonpos_c_N = []
        for cone in NormalFan(-P_new):
            cone_nonpos = cone.intersection(nonpos_orthant)
            if not cone_nonpos.is_full_dimensional():
                continue
            rep_ray = sum(vector(list(ray)) for ray in cone_nonpos.rays())
            if verbose:
                print('cone_nonpos is full dim. Performing check.')
                print(f'Representative ray is {rep_ray}.')
            if rep_ray not in nonpos_c_N:
                nonpos_c_N.append(rep_ray)

        # Get candidate sols
        mip, z_N_var = PI_N.to_linear_program(return_variable=True, solver='PPL', base_ring=QQ)
        candidate_sols = []
        for bar_c_N in nonpos_c_N:
            mip.set_objective(sum(bar_c_N[i] * z_N_var[i] for i in range(n - m)))
            mip.solve()
            z_N_opt = vector(list(mip.get_values(z_N_var).values()))
            if z_N_opt not in candidate_sols:
                candidate_sols.append(z_N_opt)
        
        return candidate_sols

    if PI.is_empty() == False:
        return[v[m:n] for v in PI.vertices()]

#     PI_N = Polyhedron(vertices=[v[m:n] for v in PI.vertices()], base_ring=QQ, backend='normaliz')

#     if PI_N.dimension() == 0:
#         return PI_N.vertices()

#     if PI_N.is_full_dimensional() == False and PI_N.dimension() > 0:
#         print('Zero Step Extend needed')
#         P_new = Extend_Polytope_To_Full_Dimension(PI_N)
#     else:
#         P_new = PI_N

#     # Get c_N's for each iteration
#     list_of_nonpos_c_N = []
#     for cone in NormalFan(-P_new):
#         cone_nonpos = cone.intersection(nonpos_orthant)
#         if not cone_nonpos.is_full_dimensional():
#             continue
#         list_of_nonpos_c_N.append(sum(vector(list(ray)) for ray in cone_nonpos.rays()))

#     mip, z_N_var = PI_N.to_linear_program(return_variable=True, solver='PPL', base_ring=QQ)
#     mip.set_max(z_N_var, bound_N)

#     z_N_list_new = []
#     for bar_c_N in list_of_nonpos_c_N:
#         mip.set_objective(sum(bar_c_N[i] * z_N_var[i] for i in range(n - m)))

#         mip.solve()
#         z_N_opt = vector(list(mip.get_values(z_N_var).values()))

#         if z_N_opt not in z_N_list_new:
#             z_N_list_new.append(z_N_opt)
# #         elif len(opt_z_N_list) > 1:
# #             print('more solutions')
# #             min_z_N = min(opt_z_N_list, key=lambda x: proximity_norm(x, A_B_inv_A_N))
# #             if min_z_N not in z_N_list_new:
# #                 z_N_list_new.append(min_z_N)
#         else:
#             print('Something went wrong. Infeasible')
#             print(g)
#             raise NotImplementedError

#     if not z_N_list_new:
#         print("Empty z_N_list")
#         raise NotImplementedError

#     return z_N_list_new

def One_Step_b_hull(A_B_inv_A_N, b1b2, s, bound_N=None, verbose=False, check_obj_c=False):
    m = len(b1b2)
    n = A_B_inv_A_N.ncols() + m

    P = Polytope_given_B1_b(A_B_inv_A_N, s, b1b2)
    PI = P.integral_hull()

    if PI.is_empty() == True:
        return []

    if check_obj_c:
        PI_N = Polyhedron(vertices=[v[m:n] for v in PI.vertices()], base_ring=QQ, backend='normaliz')
        if PI_N.dimension() == 0:
            if verbose:
                print('Dimension 0. Single Vertex Case.')
            return [v.vector() for v in PI_N.vertices()]
        else:
            if PI_N.is_full_dimensional() == False and PI_N.dimension() > 0:
                if verbose:
                    print('One Step Extend needed')
                P_new = Extend_Polytope_To_Full_Dimension(PI_N)
            else:
                if verbose:
                    print('One Step Extend unnecessary')
                P_new = PI_N
    
        # If there exists a nonpos_c_N in the cone containing a vertex, return that vertex
        nonpos_c_N = []
        for cone in NormalFan(-P_new):
            cone_nonpos = cone.intersection(nonpos_orthant)
            if not cone_nonpos.is_full_dimensional():
                continue
            rep_ray = sum(vector(list(ray)) for ray in cone_nonpos.rays())
            if verbose:
                print('cone_nonpos is full dim. Performing check.')
                print(f'Representative ray is {rep_ray}.')
            if rep_ray not in nonpos_c_N:
                nonpos_c_N.append(rep_ray)

        # Get candidate sols
        mip, z_N_var = PI_N.to_linear_program(return_variable=True, solver='PPL', base_ring=QQ)
        candidate_sols = []
        for bar_c_N in nonpos_c_N:
            mip.set_objective(sum(bar_c_N[i] * z_N_var[i] for i in range(n - m)))
            mip.solve()
            z_N_opt = vector(list(mip.get_values(z_N_var).values()))
            if z_N_opt not in candidate_sols:
                candidate_sols.append(z_N_opt)
        
        if verbose:
            print()
        
        return candidate_sols

    if PI.is_empty() == False:
#         minimal_list = minimal_vectors([v[m:n] for v in PI.vertices()])
#         nonneg_ray = [tuple(1 if j == i else 0 for j in range(n - m)) for i in range(n - m)]
#         return [vector(w) for w in Polyhedron(vertices=minimal_list, rays=nonneg_ray, backend='normaliz').vertices()]
        nonneg_ray = [tuple(1 if j == i else 0 for j in range(n - m)) for i in range(n - m)]
        return [vector(w) for w in Polyhedron(vertices=[v[m:n] for v in PI.vertices()], rays=nonneg_ray, backend='normaliz').vertices()]

#     PI_N = Polyhedron(vertices=[v[m:n] for v in PI.vertices()], base_ring=QQ, backend='normaliz')

#     if PI_N.dimension() == 0:
#         list_of_nonpos_c_N = [-ones_vector(n - m)]
#     else:
#         if PI_N.is_full_dimensional() == False and PI_N.dimension() > 0:
# #                         print('One Step Extend needed')
#             P_new = Extend_Polytope_To_Full_Dimension(PI_N)
#         else:
#             P_new = PI_N

#         # Get c_N's for each iteration
#         list_of_nonpos_c_N = []
#         for cone in NormalFan(-P_new):
#             cone_nonpos = cone.intersection(nonpos_orthant)
#             if not cone_nonpos.is_full_dimensional():
#                 continue
#             list_of_nonpos_c_N.append(sum(vector(list(ray)) for ray in cone_nonpos.rays()))

#     mip = MixedIntegerLinearProgram(maximization = True, solver = 'PPL')
#     z_all = mip.new_variable(integer = True)
#     for eqns in P.equations():
#         mip.add_constraint(sum(eqns.A()[i]*z_all[i] for i in range(n)) + eqns.b() == 0)
#     for ieqs in P.inequalities():
#         mip.add_constraint(sum(ieqs.A()[i]*z_all[i] for i in range(n)) + ieqs.b() >= 0)

#     for i in range(m,n):
#         mip.set_max(z_all[i], bound_N)
#     for bar_c_N in list_of_nonpos_c_N:
#         mip.set_objective(sum(bar_c_N[i] * z_all[m + i] for i in range(n - m)))

#         mip.solve()
#         z_opt = vector(list(mip.get_values(z_all).values()))
#         z_N_opt = z_opt[m:n]


#         obj_eq = [(-mip.get_objective_value(),) + tuple(0 for i in range(m)) + tuple(bar_c_N)]
#         bound_ineq = [(z_N_opt[i],) + tuple(-1 if j == m + i else 0 for j in range(n)) for i in range(n - m)]

#         new_Equals = P.equations_list() + obj_eq
#         new_Ineqs = P.inequalities_list() + bound_ineq

#         new_poly = Polyhedron(ieqs = new_Ineqs, eqns = new_Equals, base_ring=QQ, backend='normaliz')

#         mini_z_N_list = [z[m:n] for z in new_poly.integral_points()]

#         if len(mini_z_N_list) == 1:
#             z = mini_z_N_list[0]
#             if z not in z_N_list_new:
#                 z_N_list_new.append(z)
#         elif len(mini_z_N_list) > 1:
#             print('more solutions')
#             min_z_N = min(mini_z_N_list, key=lambda x: proximity_norm(x, A_B_inv_A_N))
#             if min_z_N not in z_N_list_new:
#                 z_N_list_new.append(min_z_N)
#         else:
#             print('Something went wrong. Infeasible')
#             raise NotImplementedError

    # Determine best (lower bound to) proximity bound
#     if not z_N_list_new:
#         print("Empty z_N")
#         raise NotImplementedError
#     return z_N_list_new

def All_Steps_b_hull(A, B, Delta=None, candidate_list=False, verbose=False, zero_step_verbose=False, check_obj_c=False):
    m = A.nrows()
    n = A.ncols()
    if not Delta:
        Delta = max(abs(x) for x in A.minors(m))
        if Delta == 1:
            return (zero_vector(n - m), 0)

    # Define A_B given basis index set B
    A_B = A.matrix_from_columns(B)

    # List of lattice points in A_B*[0,1)^m.
    UL = Unit_Lattice(A_B)

    # Define A_N corr. to A_B
    N = frozenset(range(n)).difference(B)
    A_N = A.matrix_from_columns(N)
    A_B_inv = A_B.inverse()
    A_B_inv_A_N = A_B_inv * A_N

## initial version
#     z_N_list = []

    # 0-Step
#     for g in UL:
#         # RHS for mip
#         b2 = A_B_inv*g

#         for z_N in Zero_Step_b_hull(A_B_inv_A_N, b2, bound_N = Delta - 1, verbose=verbose):
#             if z_N not in z_N_list:
#                 z_N_list.append(z_N)

#     best_z_N = max(z_N_list, key=lambda x: max(A_B_inv_A_N * x))
#     delta = max(A_B_inv_A_N * best_z_N)
#     if zero_step_verbose:
#         print(best_z_N)
#         print(f'Zero Step: delta for A_B_inv_b is {A_B_inv_A_N * best_z_N}')
#         print(z_N_list)

#     # Iterative process of One-Steps from 1 to m
#     for t in range(1, m + 1):
#         S = Subsets([0..(m - 1)], t, submultiset = False).list()

#         L = [i for i in range(int(delta + 1))]
#         b_s_list = tuple(L for i in range(t))

#         z_N_list_new = []
#         for s in S:
#             for g in UL:
#                 # b2_j = (A_B^-1 b)_j for all j except when the entries corr. to b1.
#                 b2 = A_B_inv*g
#                 for b1 in product(*b_s_list):
#                     b1b2 = copy(b2)
#                     for i in range(t):
#                         b1b2[s[i]] += b1[i]
#                     for z_N in One_Step_b_hull(A_B_inv_A_N, b1b2, s, delta, bound_N = (n - m)*Delta, verbose=verbose):
#                         if z_N not in z_N_list:
#                             z_N_list.append(z_N)
#                             z_N_list_new.append(z_N)
#         if not z_N_list_new:
#             continue
#         best_z_N = max(z_N_list_new, key=lambda x: max(A_B_inv_A_N * x))
#         delta = max(delta, max(A_B_inv_A_N * best_z_N))
#         if verbose:
#             print('New solutions found')
#             print(t)
#             print(z_N_list_new)
#             print(best_z_N)
#             print(f'One Step: delta for A_B_inv_b is {A_B_inv_A_N * best_z_N}')

    z_N_list = []
    # Enumerate for each g in UL
    for g in UL:
        # RHS for mip
        b2 = A_B_inv*g
#         if verbose:
#             print(b2)

        z_N_list_g = Zero_Step_b_hull(A_B_inv_A_N, b2, verbose=zero_step_verbose, check_obj_c=check_obj_c)

        z_N_bound = [A_B_inv_A_N * z_N - b2 for z_N in z_N_list_g]
        B_set = Set(range(m))
        delta = {B_set: [max(delta_bound[i] for delta_bound in z_N_bound) for i in range(m)]}

        if zero_step_verbose:
            print(A)
            print(A_B)
            print(g)
            print(b2)
            print(A_B_inv_A_N)
            print(z_N_bound)
            print(f'Zero Step: delta for A_B_inv_b is {delta[B_set]}')
            print(z_N_list_g)

        if max(delta[B_set]) == 0:
            if verbose:
                print(A)
                print(A_B)
                print(A_B_inv_A_N)
                print(b2)
                print(z_N_list_g)
                print('Zero step is exact')
            z_N_list = z_N_list + z_N_list_g
            continue

    # Iterative process of One-Steps from 1 to m
        for t in range(m - 1, -1, -1):
            if t == 0:
                s_c = Set()
                s = B_set
                b_s_list = []
                delta[s_c] = zero_vector(m)
                for i in s:
                    delta_i = delta[Set([i])]
                    delta_i_bound = delta_i[i]
                    for update_i in s_c:
                        if delta_i[update_i] > delta[s_c][update_i]:
                            delta[s_c][update_i] = delta_i[update_i]

                    if delta_i_bound > 1:
                        b_s_list.append([j for j in range(delta_i_bound)])
                    else:
                        b_s_list.append([0])
                for b1 in product(*b_s_list):
                    b1b2 = copy(b2)
                    for i in range(len(s)):
                        b1b2[s[i]] += b1[i]
                    z_N_list_new_b1 = One_Step_b_hull(A_B_inv_A_N, b1b2, s, verbose=verbose, check_obj_c=check_obj_c)
                    for z_N in z_N_list_new_b1:
                        if z_N not in z_N_list_g:
                            z_N_list_g.append(z_N)
                if not z_N_list_new_b1:
                    if verbose:
                        print('No new solutions')
                    continue
                if verbose:
                    print('New solutions found')
                    print(s)
                    print(b2)
                    print(b1b2)
                    print(z_N_list_new_b1)
                    print(f'One Step: delta for A_B_inv_b is {delta[s_c]}')

            else:
                S_c = Subsets(B_set, t, submultiset = False).list()
                for s_c in S_c:
                    s = B_set.difference(s_c)
                    b_s_list = []
                    delta[s_c] = zero_vector(m)
                    for i in s:
                        delta_i = delta[s_c.union(Set([i]))]
                        delta_i_bound = delta_i[i]
                        for update_i in s_c:
                            if delta_i[update_i] > delta[s_c][update_i]:
                                delta[s_c][update_i] = delta_i[update_i]

                        if delta_i_bound > 1:
                            b_s_list.append([j for j in range(delta_i_bound)])
                        else:
                            b_s_list.append([0])
                    for b1 in product(*b_s_list):
                        b1b2 = copy(b2)
                        for i in range(len(s)):
                            b1b2[s[i]] += b1[i]
                        z_N_list_new_b1 = One_Step_b_hull(A_B_inv_A_N, b1b2, s, verbose=verbose)
                        for z_N in z_N_list_new_b1:
                            if z_N not in z_N_list_g:
                                z_N_list_g.append(z_N)
                    if not z_N_list_new_b1:
                        continue
                    z_N_bound = [A_B_inv_A_N * z_N - b2 for z_N in z_N_list_new_b1]
                    for i in s_c:
                        delta_new_i = max(delta_bound[i] for delta_bound in z_N_bound)
                        if delta_new_i > delta[s_c][i]:
                            delta[s_c][i] = delta_new_i
                    if verbose:
                        print('New solutions found')
                        print(s)
                        print(b2)
                        print(b1b2)
                        print(z_N_list_new_b1)
                        print(f'One Step: delta for A_B_inv_b is {delta[s_c]}')
        z_N_list = z_N_list + z_N_list_g

    best_z_N, best_norm = max([(z_N, proximity_norm(z_N, A_B_inv_A_N)) for z_N in z_N_list], key=lambda x: x[1])
    if candidate_list:
        return (z_N_list, best_norm)
    else:
        return (best_z_N, best_norm)

def Proximity_Given_Matrix(A, Delta=None, dictionary=False, verbose=False, zero_step_verbose=False):
    if dictionary:
        big_z_N_list = dict()
    count = 0
    Basis_list = Basis_Extraction_Given_Matrix(A)
    total_basis_num = len(Basis_list)
    max_norm = 0
    for B in Basis_list:
        count = count + 1
        A_B = A.matrix_from_columns(B)
        if abs(A_B.det()) == 1 or A.ncols() == A.nrows():
            print("Basis #{} out of {} with proximity 0".format(count, total_basis_num))
            if dictionary:
                big_z_N_list[B] = (zero_vector(A.ncols() - A.nrows()), 0)
            continue
        else:
            z_N_list, z_N_norm = All_Steps_b_hull(A, B, Delta=Delta, candidate_list=False, verbose=verbose, zero_step_verbose=zero_step_verbose)
            print("Basis #{} out of {} with proximity {}".format(count, total_basis_num, z_N_norm))
            if dictionary:
                big_z_N_list[B] = (z_N_list, z_N_norm)
            else:
                if z_N_norm > max_norm:
                    max_norm = z_N_norm

    if dictionary:
        best_z_N, best_norm = max(big_z_N_list.values(), key=lambda x: x[1])
        return best_z_N, best_norm
    else:
        best_norm = max_norm
        return best_norm

def Proximity_Given_Dim_and_Delta(m, Delta, verbose=False, zero_step_verbose=False):
    prox = 0
    count = 0
    result = lattice_polytopes_with_given_dimension_and_delta(m, Delta, False)
    total_num = len(result)
    for P in result:
        A = matrix(ZZ, [v for v in P.integral_points() if v.norm(1) > 0 and next((x for x in v if x != 0), None) > 0])
        A = A.transpose()
        count = count + 1
        A_prox = Proximity_Given_Matrix(A, Delta, verbose=verbose, zero_step_verbose=zero_step_verbose)
        print("Matrix #{} of {} with proximity bound {}".format(count, total_num, A_prox))
        if A_prox >= prox:
            prox = A_prox

    return prox

def Proximity_Given_Dim_and_Delta_new(m, Delta, verbose=False, zero_step_verbose=False):
    prox = 0
    count = 0
    result = delta_classification(m, Delta, 'delta_cone')
    total_num = len(result)
    for P in result:
        A = matrix(ZZ, [v for v in P.integral_points() if v.norm(1) > 0])
        A = A.transpose()
        count = count + 1
        A_prox = Proximity_Given_Matrix(A, Delta, verbose=verbose, zero_step_verbose=zero_step_verbose)
        print("Matrix #{} of {} with proximity bound {}".format(count, total_num, A_prox))
        if A_prox >= prox:
            prox = A_prox

    return prox
