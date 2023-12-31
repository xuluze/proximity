# Proximity
Proximity test for low rank and low det

The following code runs the classification of all row rank $m$ $\Delta$-modular matrices with a trivial recession cone.
```
result = delta_classification(m, Delta, mode="delta_cone")
```
The `result` is a list of polytopes $\mathrm{conv}(\{\mathbf{0}\}\cup A)$, where $A$ is a rank $m$ $\Delta$-modular matrix.
You can also check the result from the stored database. If not exists, it will compute the classification via `delta_classification` and update the database.
```
lattice_polytopes_with_given_dimension_and_delta(m,Delta,mode="delta_cone")
```

The following code returns the proximity bound $\pi^\infty(A)$ or $\pi^1(A)$ of a given full row rank $\Delta$-modular matrix $A$ with parametric $b$ and $c$
```
prox = Proximity_Given_Matrix(A, Delta, norm='inf')
prox = Proximity_Given_Matrix(A, Delta, norm='1')
```

The following code returns the proximity bound $\pi^\infty(m, \Delta)$ or $\pi^1(m, \Delta)$ of all the rank-$m$ and $\Delta$-modular matrices.
```
prox = Proximity_Given_Dim_and_Delta_new(m, Delta, norm='inf')
prox = Proximity_Given_Dim_and_Delta_new(m, Delta, norm='1')
```