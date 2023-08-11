load("delta-classification.sage")
from itertools import product
from sage.matroids.advanced import setprint
from sage.matroids.utilities import setprint_s

def ones_vector(size):
    """
    Returns a vector in RR^size with all entries equal to 1. 
    """
    return vector([1 for i in range(size)])
    
def Delta_modularity(A):
    if A.ncols() <= A.nrows():
        A = A.transpose()

    return max(abs(x) for x in A.minors(A.nrows()))
    
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
            P = Polyhedron(vertices = eval(f_lines[n]), backend = 'normaliz') 
            AP = matrix(P.integral_points())
            L.append(AP)
    return(L)

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
    DeltaSubmatrices = []
    for B in S:
        A_B = A.matrix_from_columns(B)
        if Delta_modularity(A_B) == Delta:
            DeltaSubmatrices.append(A_B)

    return DeltaSubmatrices

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
        f_lines = f.read().splitlines()
        for i in range(len(f_lines)):
            X = matrix(eval(f_lines[i]))
            L.append(X)
    return L
