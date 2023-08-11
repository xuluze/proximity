load("delta-classification.sage")
from itertools import product
from sage.matroids.advanced import setprint
from sage.matroids.utilities import setprint_s

def ones_vector(size):
    """
    Returns a vector in RR^size with all entries equal to 1. 
    """
    return vector([1 for i in range(size)])
    
def polytopes_matrix_extraction_given_dimension_and_delta(m,Delta):
    """
    This takes all m-dimensional and Delta polytopes, obtain their integral points, encode 
    it in a matrix, and store all of these matrices in a list to be called. 
    """
    L = []
    update_delta_classification_database(m,Delta,False)
    with open(FILE_NAME_DELTA % (m,Delta),'r') as f:
        f_lines = f.read().splitlines() 
        for n in range(len(f_lines)):
            AP = matrix(eval(f_lines[n]))
            L.append(AP)
    return(L)

def collect_Delta_submatrices(A, Delta=None):
    """
    This function returns all the submatrices of a given matrix A with the same row rank and
    Delta modularity. 

    Draft: Currently brute forces to get every single submatrix.
    """
    if A.ncols() <= A.nrows():
        A = A.transpose()

    m = A.nrows()
    n = A.ncols()
    if not Delta:
        Delta = Delta_modularity(A)

    S = Subsets([0..n-1], submultiset = True).list()
    Delta_submatrices = []
    for B in S:
        A_B = A.matrix_from_columns(B)
        if Delta_modularity(A_B) = Delta:
            Delta_submatrices.append(A_B)

    return Delta_submatrices

def clean_matrix(matrix):
    """
    This takes a matrix (intended for the matrices of given dimension and Delta), 
    and returns a new matrix with no 0 rows, identity rows, or negative rows.
    """
    new_matrix = matrix[[t for t in range(matrix.nrows()//2, matrix.nrows())]]
    eliminate = []
    I = identity_matrix(new_matrix.ncols())
    O = zero_matrix(new_matrix.ncols())
    
    for i in range(new_matrix.nrows()):
        for j in range(new_matrix.ncols()):
            if new_matrix[i] == O[j]:
                eliminate.append(i)
    
    return(new_matrix.delete_rows(eliminate))
    
def clean_dim_and_delta_matrices_database(m, Delta):
    """
    This takes the matrices of the given m-dimension and Delta polytopes, cleans them according
    to the previous function, and stores them in a newly written data (.txt) file.
    """
    DeltaPolytopes = polytopes_matrix_extraction_given_dimension_and_delta(m,Delta)
    with open(FILE_NAME_DELTA_CLEAN % (m,Delta),'w') as f:
        for P in DeltaPolytopes:
            clean_P = clean_matrix(P)
            print(clean_P.rows(), file=f)
    return None

def clean_dim_and_delta_matrices(m, Delta):
    """
    This takes all m-dimensional and Delta polytopes, obtain their integral points, encode 
    it in a matrix, and store all of these matrices in a list to be called. 
    """
    clean_dim_and_delta_matrices_database(m, Delta)
    L = []
    with open(FILE_NAME_DELTA_CLEAN % (m,Delta),'r') as f:
        f_lines = f.read().splitlines() 
        for n in range(len(f_lines)):
            CAP = matrix(eval(f_lines[n]))
            L.append(CAP)
    return(L)

def full_dim_and_delta_matrices_database(m,Delta):
    MaxDeltaMatrices = clean_dim_and_delta_matrices(m,Delta)
    with open(FILE_NAME_DELTA_FULL % (m,Delta),'w') as f:
        for A in MaxDeltaMatrices:
            print(A.rows(), file=f)
            DeltaSubmatrices = collect_Delta_submatrices(A, Delta)                                
            for B in DeltaSubmatrices:       
                print(B.rows(), file=f)
    return None

def full_dim_and_delta_matrices(m,Delta):
    full_dim_and_delta_matrices_database(m,Delta)
    L = []
    with open(FILE_NAME_DELTA_FULL % (m,Delta), 'r') as f:
        f_lines = f.read.splitlines()
        for i in range(len(f_lines)):
            X = matrix(eval(f_lines[i]))
            L.append(X)
    return L
            
def IP_and_LP_Given_Matrix_and_Constraints(A, b1, b2, d1, d2, c):
    """
    This function takes in constraint matrix A, LHS and RHS vectors b1 and b2,
    boundary vectors d1 and d2, and objective function c and solves the IP and 
    corr. LP relaxation with the feasible region P(A,b1,b2,d1,d2,c). It displays 
    the info of the IP and LP, along with the optimal solutions and their infinity 
    norm distance.
    """
    # Integer Linear Program corr. to P(A,b1,b2,c,d1,d2)
    ip = MixedIntegerLinearProgram(maximization = True, solver = 'PPL') # max IP
    z = ip.new_variable(integer = True) # integer inputs
    ip.add_constraint(-A*z <= -b1) # Az => b1
    ip.add_constraint(A*z <= b2) # Az <= b2
    for i in range(A.ncols()):
        ip.add_constraint(z[i] <= d2[i]) # z <= d2
        ip.add_constraint(d1[i] <= z[i]) # z => d1
    ip.set_objective(sum(c[i]*z[i] for i in range(A.ncols()))) # max{c*z}
    
    # Get optimal solution as a vector
    ip.solve()
    L1 = list(ip.get_values(z).values())
    z_opt = vector(L1)
    
    # Linear Program corr. to P(A,b1,b2,c,d1,d2)
    lp = MixedIntegerLinearProgram(maximization = True, solver = 'PPL') # max IP
    x = lp.new_variable(real = True) # real inputs
    lp.add_constraint(-A*x <= -b1) # Ax => b1
    lp.add_constraint(A*x <= b2) # Ax <= b2
    for i in range(A.ncols()):
        lp.add_constraint(x[i] <= d2[i]) # x <= d2
        lp.add_constraint(d1[i] <= x[i]) # x => d1
    lp.set_objective(sum(c[i]*x[i] for i in range(A.ncols()))) # max{c*x}
    
    # Get optimal solution as a vector
    lp.solve()
    L2 = list(lp.get_values(x).values())
    x_opt = vector(L2)
    
    # Calculate Infinity Norm Between opt. LP sol. and opt. IP sol.
    y = x_opt - z_opt 
    inf_norm = y.norm(Infinity)
    
    # Return Optimization Info
    return(z_opt,x_opt,inf_norm)
        
def Polyhedra_Of_Optimal_Integer_Solutions(A, b1, b2, d1, d2, c, z_opt):
    """
    This function creates the polyhedron of all integer points z satisfying
    P(A,b1,b2,d1,d2) that attain c*z = c*z_opt. Thus, it gathers all of the 
    optimal integer solutions of P(A,b1,b2,d1,d2).
    """
    b1_matrix = matrix(b1)
    b2_matrix = matrix(b2)
    d1_matrix = matrix(d1)
    d2_matrix = matrix(d2)
    AT = A.transpose()
    
    I = identity_matrix(A.ncols())
    A_b2_ineq = b2_matrix.stack(-AT).transpose() 
    A_b1_ineq = (-b1_matrix).stack(AT).transpose()
    d1_ineq = (-d1_matrix).stack(I).transpose()
    d2_ineq = d2_matrix.stack(-I).transpose()
    
    obj_eq = [-c*z_opt]
    for i in range(len(c)):
        obj_eq.append(c[i])
        
    H_ineq = A_b2_ineq.stack(A_b1_ineq).stack(d2_ineq).stack(d1_ineq)
    H_eq = matrix(obj_eq)
    
    P = Polyhedron(ieqs = H_ineq, eqns = H_eq) 
    
    return(P)
    
def Get_Minimum_Infinity_Norm_Given_Polyhedron_and_LP_Opt(P, x_opt):
    """
    This function takes in a Polyhedron and a vector x_opt corr. to an optimal
    solution to some LP. It calculates the minimum distance between x_opt and a 
    point in P, and returns the vector attaining the minimum and the value of the
    minimum itself.
    """
    PI = P.integral_points()
    z_min = PI[0]
    a = x_opt - z_min
    min_norm =  a.norm(Infinity)
    
    for z in P.integral_points():
        y = x_opt - z
        inf_norm = y.norm(Infinity)
        if inf_norm <= min_norm:
            min_norm = inf_norm
            z_min = z
    return(z_min, min_norm)
    
def Proximity_Dependent_of_Everything(A,b1,b2,d1,d2,c):
    """
    This function takes in all LP and IP information (constraint matrix A,
    RHS vectors b1, b2, boundary vectors d1, d2, and an objective function c)
    and calculates the proximity bound between the LP and IP problems corr. to 
    the inputs. It returns the LP optimal solution, the closest IP optimal solution
    and their distance in infinity norm.
    """
    L = IP_and_LP_Given_Matrix_and_Constraints(A,b1,b2,d1,d2,c)
    z_opt = L[0]
    x_opt = L[1]
    inf_norm = L[2]
    
    # Check if x is integer
    round_x_opt = vector([floor(x_opt[i]) for i in range(len(x_opt))])
    
    if (x_opt - round_x_opt).norm(Infinity) < 1e-10:
        min_norm = 0
        z_min = x_opt
        return(z_min, x_opt, min_norm)
    # If x is not integer, obtain the closest integer optimal solution
    else:
        P = Polyhedra_Of_Optimal_Integer_Solutions(A, b1, b2, d1, d2, c, z_opt)
        M = Get_Minimum_Infinity_Norm_Given_Polyhedron_and_LP_Opt(P, x_opt)
        z_min = M[0]
        min_norm = M[1]
        return(z_min, x_opt, min_norm)

def Big_Basis_Extraction_Given_Matrix(A):
    """
    This function takes a matrix A and determines all bases of the matrix
    [A -A I -I] to get every possible basis in considering the LP problem
    [A -A I -I]x <= [b2 b1 d2 d1].
    """
    I = identity_matrix(A.ncols())
    BigA = A.stack(-A).stack(I).stack(-I).transpose()
    M = Matroid(matrix = BigA)
    
    X = list(M.bases())
    L = eval(setprint_s(X))
    
    Bases = []
    
    for set in L:
        s = [i for i in set]
        BigAB = BigA.matrix_from_columns(s)
        Bases.append(BigAB)
    
    return(Bases)
    
def Polytope_given_Matrix_RHS_and_Boundary(A, b1, b2, d1, d2):
    """
    This function creates the polytope P(A,b1,b2,d1,d2) corr. to
    b1 <= Ax <= b2 and d1 <= x <= d2.
    """
    b1_matrix = matrix(b1)
    b2_matrix = matrix(b2)
    d1_matrix = matrix(d1)
    d2_matrix = matrix(d2)
    AT = A.transpose()
    
    I = identity_matrix(A.ncols())
    A_b2_ineq = b2_matrix.stack(-AT).transpose() 
    A_b1_ineq = (-b1_matrix).stack(AT).transpose()
    d1_ineq = (-d1_matrix).stack(I).transpose()
    d2_ineq = d2_matrix.stack(-I).transpose()
        
    H_ineq = A_b2_ineq.stack(A_b1_ineq).stack(d2_ineq).stack(d1_ineq)
    
    P = Polyhedron(ieqs = H_ineq)
    return(P)

def integral_hull(P):
    """
    This function returns the integral hull of the polytope entered.
    """
    V = P.integral_points()
    PI = Polyhedron(vertices = V)
    
    return(PI)
    
def Extend_Polytope_To_Full_Dimension(P):
    """
    This function takes a polytope and adds width in the appropriate
    dimensions such that the resulting polytope is full dimensional.
    If a full dimensional polytope is entered, it is returned.
    """
    L = []
    
    for i in P.equations():
        i_new_lb_ext = list(i.vector())
        i_new_lb_ext[0] = i.vector()[0] + 1
        i_new_ub_ext = list(-i.vector())
        i_new_ub_ext[0] = -i.vector()[0] + 1
        L.append(i_new_lb_ext)
        L.append(i_new_ub_ext)
        
    for i in P.inequalities_list():
        L.append(i)
        
    P_new = Polyhedron(ieqs = L)
    
    return(P_new)
    
def list_of_objectives_by_normal_fan_given_polytope(P):
    """
    This function takes a polytope P and obtains the common refinement
    of the normal fan of P and integer-hull(P), adding width/considering
    projections in lower dimensions if necessary to obtain full-dimension.
    Then, for each cone in the common refinement, picks a representative
    objective function, and returns all in a list.
    """
    PI = integral_hull(P)
    
    if P.dimension() == 0:
        return([ones_vector(P.ambient_dimension())]) 
    else:
        if P.is_full_dimensional() == False:
            P_new = P.affine_hull_projection()
            PI = integral_hull(P_new)
        else:
            P_new = P

        if PI.is_full_dimensional() == False:
            PI_new = Extend_Polytope_To_Full_Dimension(PI)
        else:
            PI_new = PI

        P_fan = NormalFan(P_new)
        PI_fan = NormalFan(PI_new)
        Ref_fan = P_fan.common_refinement(PI_fan)
        list_of_obj = []
        for cone in Ref_fan:
            list_of_rays = []
            for ray in cone.rays():
                ray_vec = vector(list(ray))
                list_of_rays.append(ray_vec)

            c = sum(i for i in list_of_rays)
            list_of_obj.append(c)

        return(list_of_obj)

def Proximity_Indep_of_Objective(A,b1,b2,d1,d2):
    """
    This function takes in all Polyhedral information (constraint matrix A,
    RHS vectors b1, b2, and boundary vectors d1, d2) and determines the maximum
    possible proximity, enumerating every possible objective function.
    """
    P = Polytope_given_Matrix_RHS_and_Boundary(A, b1, b2, d1, d2)    
    list_of_obj = list_of_objectives_by_normal_fan_given_polytope(P)
            
    x_best = zero_vector(A.ncols())
    z_best = zero_vector(A.ncols())
    inf_norm = 0
            
    for c in list_of_obj:
        if len(c) != A.ncols():
            diff = A.ncols() - len(c)
            for i in range(diff):
                L = list(c)
                L.append(0)
                c = vector(L)
        
        Prox = Proximity_Dependent_of_Everything(A,b1,b2,d1,d2,c)
        z_opt = Prox[0]
        x_opt = Prox[1]
        norm = Prox[2]
        if norm > inf_norm:
            inf_norm = norm
            z_best = z_opt
            x_best = x_opt
        else:
            continue
            
    return(z_best, x_best, inf_norm)

def Basis_Extraction_Given_Matrix(A):
    """
    Given a matrix A, this function determines all of the maximal-sized invertible
    square submatrices of A and returns them in a list.
    """
    if A.ncols() >= A.nrows():
        A_new = A
        A_mat = Matroid(matrix = A_new)
        X = list(A_mat.bases())
        L = eval(setprint_s(X))    
        Bases = []
        for S in L:
            B = [i for i in S]
            Bases.append(B)
    else:
        A_new = A.transpose()
        A_mat = Matroid(matrix = A_new)
        X = list(A_mat.bases())
        L = eval(setprint_s(X))    
        Bases = []
        for S in L:
            B = [i for i in S]
            Bases.append(B)
    
    return(Bases)

def Unit_Lattice(A):
    """
    This function returns, in a list, all of the vectors in the
    lattice A*[0,1)^n. (A is n by n.) 
    """
    D, U, V = A.smith_form()
    A_diagonal = vector(D.diagonal())
    max_1_norm = A_diagonal.norm(1) - len(A_diagonal)
    
    iv = []
    for a in range(max_1_norm + 1):
        for x in IntegerVectors(a,len(A_diagonal)):
            iv.append(x)
    
    L = [vector(i) for i in iv]
    eliminate = []
    for i_vec in L:
        for i in range(len(A_diagonal)):
            if i_vec[i] >= A_diagonal[i]:
                eliminate.append(i_vec)
                
    UL = []
    for i in L:
        if i not in eliminate and i.norm(1) >= 1:
            UL.append(U.inverse()*i)
            
    return(UL)
    
def Zero_Step_b_hull(A,B):
    # Compute Delta(A) (largest basis-subdeterminant in abs value.)
    if A.ncols() <= A.nrows():
        Delta = max(abs(x) for x in A.minors(A.ncols()))
    else:
        Delta = max(abs(x) for x in A.minors(A.nrows()))
    
    # Define A_B given basis index set B
    A_B = A.matrix_from_columns(B)
    
    # List of lattice points in A_B*[0,1)^m.
    UL = Unit_Lattice(A_B)

    # Define A_N corr. to A_B
    N = []
    for t in range(A.ncols()):
        if t not in B:
            N.append(t)
    A_N = A.matrix_from_columns(N)
    
    z_N_list_new = []
    for g in UL:
        # RHS for mip 
        b2 = A_B.inverse()*g
        
        # Constraint matrix for mip
        N_2 = A_B.inverse()*A_N
        
        mip = MixedIntegerLinearProgram(maximization = True, solver = 'PPL')
        w = mip.new_variable(integer = True)
        z_N = mip.new_variable(integer = True, nonnegative = True)
        mip.add_constraint(identity_matrix(N_2.nrows())*w + N_2 * z_N == b2)
        
        P = mip.polyhedron().change_ring(QQ)

        if P.is_empty() == True:
            continue

        if P.is_compact() == False:
            a = Delta - 1
            # mip.set_max(w, a)
            mip.set_max(z_N, a)

        P = integral_hull(mip.polyhedron().change_ring(QQ))

        if P.is_empty() == True:
            continue

        if P.dimension() == 0:    
            list_of_nonpos_c_N = [-ones_vector(N_2.ncols())]
        else:
            if P.is_full_dimensional() == False and P.dimension() > 0:
                P_new = Extend_Polytope_To_Full_Dimension(P)
            else:
                P_new = P

            # Get c_N's for each iteration
            C = []
            for cone in NormalFan(P_new):
                list_of_rays = []
                for ray in cone.rays():
                    ray_vec = vector(list(ray))
                    list_of_rays.append(ray_vec)

                c = sum(i for i in list_of_rays)
                C.append(c)

            list_of_c_N = []
            for c in C:
                c_N = vector([c[i] for i in range(len(b2), len(b2) + N_2.ncols())])
                if c_N not in list_of_c_N:
                    list_of_c_N.append(c_N)
                
            remove_c_N = []
            for c_N in list_of_c_N:
                for i in range(len(c_N)):
                    if c_N[i] > 0:
                        remove_c_N.append(c_N)
                        break

            list_of_nonpos_c_N = []
            for c_N in list_of_c_N:
                if c_N not in remove_c_N:
                    list_of_nonpos_c_N.append(c_N)
        
        for bar_c_N in list_of_nonpos_c_N:
            mip.set_objective(sum(bar_c_N[i]*z_N[i] for i in range(len(bar_c_N))))

            mip.solve()
            mip_poly = mip.polyhedron()
            opt_values = list(mip.get_values(z_N).values())
            z_N_opt = vector(opt_values)

            obj_eq = [-bar_c_N*z_N_opt]
            for i in range(len(b2)):
                obj_eq.append(0)
            for i in range(len(bar_c_N)):
                obj_eq.append(bar_c_N[i])

            z_N_opt_matrix = matrix(z_N_opt)
            
            bound_ineq = z_N_opt_matrix.stack(zero_matrix(len(b2), N_2.ncols())).stack(-identity_matrix(N_2.ncols())).transpose()
            nonneg_ineq = zero_matrix(1, N_2.ncols()).stack(zero_matrix(len(b2), N_2.ncols())).stack(identity_matrix(N_2.ncols())).transpose()
            
            Ineqs = []
            Equals = []
            for i in mip_poly.inequalities():
                i_vec = list(i.vector())
                Ineqs.append(i_vec)

            for e in mip_poly.equations():
                e_vec = list(e.vector())
                Equals.append(e_vec)

            for i in bound_ineq.rows():
                Ineqs.append(list(i))

            for i in nonneg_ineq.rows():
                Ineqs.append(list(i))

            Equals.append(obj_eq)

            new_poly = Polyhedron(ieqs = Ineqs, eqns = Equals).change_ring(QQ)
            
            opt_z_N_list = []
            for i in new_poly.integral_points():
                extract_z_N = vector([i[a] for a in range(len(b2), len(b2) + N_2.ncols())])
                opt_z_N_list.append(extract_z_N)
                
            if len(opt_z_N_list) > 0:
                min_z_N = opt_z_N_list[0]
                min_norm = max([min_z_N.norm(Infinity), (A_B.inverse()*A_N*min_z_N).norm(Infinity)])
                for z in opt_z_N_list:
                    z_norm = max([z.norm(Infinity), (A_B.inverse()*A_N*z).norm(Infinity)])
                    if z_norm <= min_norm:
                        min_norm = z_norm
                        min_z_N = z
                
                if min_z_N not in z_N_list_new:
                    z_N_list_new.append(min_z_N)
            else:
                print('Something went wrong. Infeasible')
                print(g)
                continue
    
    best_z_N = z_N_list_new[0]
    best_norm = max([best_z_N.norm(Infinity), (A_B.inverse()*A_N*best_z_N).norm(Infinity)])
    for z_N in z_N_list_new:
        z_N_norm = max([z_N.norm(Infinity), (A_B.inverse()*A_N*z_N).norm(Infinity)])
        if z_N_norm >= best_norm:
            best_norm = z_N_norm
            best_z_N = z_N
            
    print(f'delta for b: {A_B.inverse()*A_N*best_z_N}')
    print(z_N_list)
    delta = (A_B.inverse()*A_N*best_z_N).norm(Infinity)
    return(delta, z_N_list_new)
    
def One_Step_b_hull(A,B,t,delta):
    # Compute Delta(A) (largest basis-subdeterminant in abs value.)
    if A.ncols() <= A.nrows():
        Delta = max(abs(x) for x in A.minors(A.ncols()))
    else:
        Delta = max(abs(x) for x in A.minors(A.nrows()))
    
    # List to store all min-distance achieving z_N's.
    z_N_list_new = []
    
    A_B = A.matrix_from_columns(B)
    
    # Set of values for b1 (components of b in which (A_B^-1 b)_i <= ||A_B^-1A_Nz_N||_inf)
    Delta_A_B = abs(A_B.det())
    L = [i/Delta_A_B for i in range(int(Delta_A_B*delta + 1))]
    
    # List of lattice points in A_B*[0,1)^m.
    UL = Unit_Lattice(A_B)

    # Define A_N corr. to A_B
    N = []
    for i in range(A.ncols()):
        if i not in B:
            N.append(i)
    A_N = A.matrix_from_columns(N)
    
    S = Subsets([0..(A_B.nrows() - 1)], t, submultiset = True).list()
    
    b_s_list = tuple(L for i in range(t))
    
    for s in S:
        for b1v in product(*b_s_list):
            b1 = vector(b1v)
            for g in UL: 
                # b2_j = (A_B^-1 b)_j for all j except when the entries corr. to b1.
                b2 = A_B.inverse()*g
                for i in range(len(b2)):
                    if i in S:
                        b2[i] = 0

                # N_1 = A_B^-1 A_N only on the entries corr. to b1.
                N_1 = A_B.inverse()*A_N
                keep = []
                for i in range(N_1.nrows()):
                    if i in s:
                        keep.append(i)
                N_1 = N_1.matrix_from_rows(keep)

                # N_2 = A_B^-1 A_N, except 0 on the row corr. to the entries of b1.
                N_2 = A_B.inverse()*A_N
                for i in range(N_2.nrows()):
                    if i in S:
                        N_2[i] = zero_vector(N_2.ncols())

                # For each c_N, obtain a min-distance z_N and add it to the list.
                mip = MixedIntegerLinearProgram(maximization = True, solver = 'PPL')
                w = mip.new_variable(integer = True, nonnegative = True)
                y = mip.new_variable(integer = True)
                z_N = mip.new_variable(integer = True, nonnegative = True)
                mip.add_constraint(identity_matrix(N_1.nrows())*w + N_1 * z_N == b1)
                mip.add_constraint(identity_matrix(N_2.nrows())*y + N_2 * z_N == b2)

                P = mip.polyhedron().change_ring(QQ)

                if P.is_empty() == True:
                    continue

                if P.is_compact() == False:
                    a = (A.ncols() - A.nrows())*Delta
                    mip.set_max(w, a)
                    mip.set_max(z_N, a)
                    mip.set_max(y, a)

                P = integral_hull(mip.polyhedron().change_ring(QQ))

                if P.is_empty() == True:
                    continue

                if P.dimension() == 0:    
                    list_of_nonpos_c_N = [-ones_vector(N_1.ncols())]
                else:
                    if P.is_full_dimensional() == False and P.dimension() > 0:
                        P_new = Extend_Polytope_To_Full_Dimension(P)
                    else:
                        P_new = P

                    # Get c_N's for each iteration
                    C = []
                    for cone in NormalFan(P_new):
                        list_of_rays = []
                        for ray in cone.rays():
                            ray_vec = vector(list(ray))
                            list_of_rays.append(ray_vec)

                        c = sum(i for i in list_of_rays)
                        C.append(c)

                    list_of_c_N = []
                    for c in C:
                        c_N = vector([c[i] for i in range(N_1.nrows(), N_1.nrows() + N_1.ncols())])
                        if c_N not in list_of_c_N:
                            list_of_c_N.append(c_N)

                    remove_c_N = []
                    for c_N in list_of_c_N:
                        for i in range(len(c_N)):
                            if c_N[i] > 0:
                                remove_c_N.append(c_N)
                                break

                    list_of_nonpos_c_N = []
                    for c_N in list_of_c_N:
                        if c_N not in remove_c_N:
                            list_of_nonpos_c_N.append(c_N)

                for bar_c_N in list_of_nonpos_c_N:
                    mip.set_objective(sum(bar_c_N[i]*z_N[i] for i in range(len(bar_c_N))))
                    
                    mip.solve()
                    mip_poly = mip.polyhedron()
                    opt_values = list(mip.get_values(z_N).values())
                    z_N_opt = vector(opt_values)

                    obj_eq = [-bar_c_N*z_N_opt]
                    for i in range(len(b1)):
                        obj_eq.append(0)
                    for i in range(len(bar_c_N)):
                        obj_eq.append(bar_c_N[i])
                    for i in range(len(b2)):
                        obj_eq.append(0)

                    z_N_opt_matrix = matrix(z_N_opt)
                                        
                    bound_ineq = z_N_opt_matrix.stack(zero_matrix(len(b1), N_1.ncols())).stack(-identity_matrix(N_1.ncols())).stack(zero_matrix(len(b2), N_1.ncols())).transpose()
                    nonneg_ineq = zero_matrix(1, len(b1) + N_1.ncols()).stack(identity_matrix(len(b1) + N_1.ncols())).stack(zero_matrix(len(b2), len(b1) + N_1.ncols())).transpose()

                    Ineqs = []
                    Equals = []
                    for i in mip_poly.inequalities():
                        i_vec = list(i.vector())
                        Ineqs.append(i_vec)

                    for e in mip_poly.equations():
                        e_vec = list(e.vector())
                        Equals.append(e_vec)

                    for i in bound_ineq.rows():
                        Ineqs.append(list(i))

                    for i in nonneg_ineq.rows():
                        Ineqs.append(list(i))

                    Equals.append(obj_eq)

                    new_poly = Polyhedron(ieqs = Ineqs, eqns = Equals).change_ring(QQ)

                    mini_z_N_list = []
                    for i in new_poly.integral_points():
                        extract_z_N = vector([i[a] for a in range(len(b1), len(b1) + N_1.ncols())])
                        mini_z_N_list.append(extract_z_N)
                    
                    if len(mini_z_N_list) > 0:
                        min_z_N = mini_z_N_list[0]
                        min_norm = max([min_z_N.norm(Infinity), (A_B.inverse()*A_N*min_z_N).norm(Infinity)])
                        for z in mini_z_N_list:
                            z_norm = max([z.norm(Infinity), (A_B.inverse()*A_N*z).norm(Infinity)])
                            if z_norm <= min_norm:
                                min_norm = z_norm
                                min_z_N = z

                        if min_z_N not in z_N_list_new:
                            z_N_list_new.append(min_z_N)
                    else:
                        print('Something went wrong. Infeasible')
                        print(g)
                        continue

    # Determine best (lower bound to) proximity bound
    best_z_N = z_N_list_new[0]
    best_norm = max([best_z_N.norm(Infinity), (A_B.inverse()*A_N*best_z_N).norm(Infinity)])
    for z_N in z_N_list_new:
        z_N_norm = max([z_N.norm(Infinity), (A_B.inverse()*A_N*z_N).norm(Infinity)])
        if z_N_norm >= best_norm:
            best_norm = z_N_norm
            best_z_N = z_N
    
    print(f'delta for b: {A_B.inverse()*A_N*best_z_N}')
    print(z_N_list_new)
    delta_new = max(delta, (A_B.inverse()*A_N*best_z_N).norm(Infinity))
    return(delta_new, z_N_list_new)
    
def All_Steps_b_hull(A,B):    
    z_N_list = []
    if A.ncols() <= A.nrows():
        A = A.transpose()
    
    # 0-Step
    delta, z_N_list_zero = Zero_Step_b_hull(A,B)
    for z_N in z_N_list_zero:
        # z_N.set_immutable()
        if z_N not in z_N_list:
            z_N_list.append(z_N)
    
    # Iterative process of One-Steps from 1 to m
    for t in range(1,A.nrows()+1):
        delta, z_N_list_new = One_Step_b_hull(A, B, t, delta)
        for z_N in z_N_list_new:
            # z_N.set_immutable()
            if z_N not in z_N_list:
                z_N_list.append(z_N)
    
    return(z_N_list)
