import numpy as np

# MATRICES
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)

# matrix has m x n real values ; where m is the number of rows and n is the number of columns , all spread along a reactangular scheme in a ordered way

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# MATRICE MULTIPLICATION AND ADDITION

# addition

A = np.array([[1, 2], [3, 4]])
B = np.array([[5,6],[7,8]])
C = A + B
print(C)

# for addition both matrices must be of same order

# multiplication

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[5,6],[7,8]])
C = A @ B # or np.dot(A,B)
print(C)

# for multiplication the number of columns in the first matrix must be equal to the number of rows in the second matrix

# MATRIX MULTIPLICATION IS NOT COMMUTATIVE

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = A @ B
D = B @ A
print(C)
print(D)
# you will see that C is not equal to D hence matrix multiplication is not commutative aka AB != BA

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# IDENTITY MATRIX
I = np.eye(3)
print(I)

# identity matrix is a square matrix (with equal rows and columns) that has 1's along the main diagonal and 0's everywhere else

# SOME OTHER PROPERTIES
# 1. A + B = B + A
# 2. A + (B + C) = (A + B) + C
# 3. A + 0 = A
# 4. A + (-A) = 0
# 5. A(B + C) = AB + AC
# 6. (A + B)C = AC + BC
# 7. A(I) = A

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# INVERSE OF A MATRIX

# let there be a matrix A (square) then if another matrix B of same order exists such that AB = BA = I then B is the inverse of A and vice versa
# not every matrix has an inverse
A = np.array([[1,2],[3,4]])
B = np.linalg.inv(A)
print(B)

# a good way to check if matrix has inverse or not is to find its determinant and if it is not equal to 0 then it has an inverse
det = np.linalg.det(A)
print(det)

# TRANSPOSE OF A MATRIX

# let there be a matrix A then the transpose of A is a matrix B such that B[i][j] = A[j][i] ( where i and j are rows and columns respectively)
A = np.array([[1,2],[3,4],[5,6]])
B = A.T
print(B)

# another way to represent transpose is to place a T as an exponenet to the matrix like B = A^T

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# SYMMETRIC MATRIX
# a matrix is symmetric if A = A^T
A = np.array([[1,2],[2,1]])
B = A.T
if np.array_equal(A,B):
    print("Symmetric")
else:
    print("Not Symmetric")

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# MULTIPLICATION BY SCALAR
"""
for multiplciation by scalar the following holds
- associativity : a(bA) = abA
- distributive : (a+b)A = aA + bA
"""
# example of distributivity using scalar multiplication
A = np.array([[1,2],[3,4]])
B = 2*A
C = A+A
print(B)
print(C)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# REPRESENTATION OF SYSTEM OF LINEAR EQUATIONS

"""
consider a system of linear equations
2x + 3y + 5z = 1
4x - 2y - 7z = 8
9x + 5y - 3z = 2
"""
# this can be represented as AX = B where A is the matrix of coefficients , X is the matrix of variable and B is the matrix of constants
A = np.array([[2,3,5],[4,-2,-7],[9,5,-3]])
B = np.array([[1],[8],[2]])
X = np.linalg.solve(A,B)
print(X)

# this can return problems if the matrix A is singular (i.e. it does not have an inverse) in that case we can use the pseudo inverse
X = np.linalg.pinv(A) @ B
print(X)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# PARTICULAR AND GENERAL SOLUTION  
# A particular solution is a specific solution to the system AX = B.  
# The general solution consists of a particular solution plus all solutions to AX = 0 (the null space of A).  
# If A is singular (dependent rows), the system has infinitely many solutions, and the general solution is:  
# X = X_particular + t * X_homogeneous (where t is any real number).

# Matrix A (coefficients) and B (constants)
A = np.array([[1, 2, -1], [2, 4, -2], [3, 6, -3]])  # singular (bad) matrix
B = np.array([[3], [6], [9]])

# Find one valid answer (particular solution)
X_particular = np.linalg.pinv(A) @ B  # Works even if A is bad

# Find extra solutions that do nothing (homogeneous solution)
U, S, Vt = np.linalg.svd(A)  # Magic function to find "extra solutions"
X_homogeneous = Vt.T[:, S < 1e-10]  # Extract them

# Show 10 possible answers by picking t = -5 to 5
t_values = np.arange(-5, 6)
X_general = [X_particular + t * X_homogeneous for t in t_values]

# Print results
print("One valid answer (particular solution):\n", X_particular)
print("Extra solutions that do nothing (homogeneous part):\n", X_homogeneous)
print("All possible answers (general solution):")
for i, X in enumerate(X_general):
    print(f"t = {t_values[i]}:\n{X}")

# in all honesty this is a bit too much to understand in one go but the general idea is that if the matrix is singular then the system has infinitely many solutions , also general solution is much less used and mostly particular solution is used

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# ELEMENTARY OPERATIONS

# now numpy can automatically solve the system of linear equations but if you want to do it manually then you can use elementary operations
# here is an attempt at showing how its done

# manually

A = np.array([[2, 3, -1], [4, 5, -3], [-2, -3, 2]], dtype=float)  # Use float for division

print("Original Matrix:\n", A)

# 1. Swap row 1 and row 2 (R1 <-> R2)
A[[0, 1]] = A[[1, 0]]
print("\nAfter Swapping Row 1 and Row 2:\n", A)

# 2. Multiply row 2 by 0.5 (R2 -> 0.5 * R2)
A[1] = A[1] * 0.5
print("\nAfter Multiplying Row 2 by 0.5:\n", A)

# 3. Add -2 times row 1 to row 3 (R3 -> R3 - 2*R1)
A[2] = A[2] - 2 * A[0]
print("\nAfter Row Operation R3 -> R3 - 2*R1:\n", A)

# automatically

A = np.array([[2, 3, -1], [4, 5, -3], [-2, -3, 2]], dtype=float)  # Use float for division
B = np.array([[1], [8], [2]], dtype=float)
X = np.linalg.solve(A, B)  # Directly solves AX = B
print("Solution using np.linalg.solve:\n", X)

# the elementary operations are used to convert the matrix into a upper triangular matrix and then the solution can be found by back substitution
# also known as gauss elimination method

X = np.linalg.pinv(A) @ B  # Works even if A is singular
print("Solution using pseudo-inverse:\n", X)


# a matrix is in row echelon form if
"""
- all rows consisting of only zeros are at the bottom
- the first non zero element in a row is 1
- the first non zero element in a row is to the right of the first non zero element in the row above it
- Looking at nonzero rows only, the first nonzero number from the left (also called the pivot or the leading coefficient) is always strictly to the right of the pivot of the row above it.
- in simple words : the matrix is upper triangular and the first non zero element in a row is 1
"""

# pivot means the first non zero element in a row
# reduced row echelon form
# a matrix is in reduced row echelon form if
"""
- it is in row echelon form
- every pivot is 1
- the pivot is only nonzero entry in its column
"""

# representation of a matrix in reduced row echelon form

A = np.array([[1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 1]])
print(A)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# calculating the inverse

# numpy offers a direct way so i wont be going the gaussian elimination way ; you can read on it if you want online
A = np.array([[1,2],[3,4]])
B = np.linalg.inv(A)
print(B)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# VECTOR SPACES

# a vector space is a set of vectors that is closed under two operations : addition and scalar multiplication

# GROUPS

# symbol definition ; ⊗
# if A is an m x n matrix and B is a p x q matrix then their kronecker product is an mp x nq matrix which is represented as A ⊗ B

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

K = np.kron(A, B)  # Kronecker product
print(K)

# back to groups

# Explanation of symbols:
# - G: A set (a collection of elements)
# - ⊗: A binary operation (can be addition, multiplication, etc.)  
# - G × G → G: This means ⊗ takes two elements from G and outputs another element in G.
# - ∀: "For all" (universal quantifier)
# - ∈: "Belongs to" (membership in a set)
# - ∃: "There exists" (existential quantifier)
# - e: The identity (neutral) element in G
# - x⁻¹: The inverse of x (an element in G that undoes x under ⊗)

# Definition  (Group)
# A group is a set G with an operation ⊗ : G × G → G that satisfies:

# 1. Closure: ∀x, y ∈ G, the result of x ⊗ y is also in G.
#    (i.e., G is closed under the operation ⊗)

# 2. Associativity: ∀x, y, z ∈ G, we have (x ⊗ y) ⊗ z = x ⊗ (y ⊗ z).
#    (i.e., the operation ⊗ is associative)

# 3. Neutral element (identity element): There exists an element e ∈ G
#    such that ∀x ∈ G: x ⊗ e = x and e ⊗ x = x.
#    (i.e., e acts as an identity for all elements in G)

# 4. Inverse element: For every x ∈ G, there exists an element y ∈ G
#    such that x ⊗ y = e and y ⊗ x = e, where e is the identity element.
#    (i.e., every element has an inverse, often written as x⁻¹)

# in simple words a group is a set of elements with an operation that satisfies the above 4 conditions

# numpy implementation of all four conditions :

G = np.array([0, 1, 2, 3, 4])  # A set of elements
mod = 5 # Modulo operation : why : because we are working with a cyclic group : what that means : if you go beyond the last element then you come back to the first element

# 1. closure:
# ⊗: Modulo operation
closure = all((x+y)%mod in G for x in G for y in G) # for all x and y in G the result of x ⊗ y is also in G
print("Closure:", closure)

# 2. associativity : 
associativity = all(((x+y)%mod+z)%mod == (x+(y+z)%mod)%mod for x in G for y in G for z in G) # for all x, y, z in G, we have (x ⊗ y) ⊗ z = x ⊗ (y⊗z)
print("Associativity:", associativity)

# 3. Neutral element: Find e such that x ⊗ e = x for all x ∈ G
identity = [e for e in G if all((x + e) % mod == x for x in G)] # there exists an element e ∈ G such that ∀x ∈ G: x ⊗ e = x and e ⊗ x = x
print("Neutral element (identity):", identity)  # Should print [0]

# 4. Inverse element: Find y such that x ⊗ y = e for all x ∈ G
inverse_elements = {x: [y for y in G if (x + y) % mod == 0] for x in G} # for every x ∈ G, there exists an element y ∈ G such that x ⊗ y = e and y ⊗ x = e, where e is the identity element
print("Inverse elements:", inverse_elements)  

# to sum all this up in simple words : a group is a set of elements with an operation that satisfies the above 4 conditions ; the modulo operation is a good example of a group

"""
Closure: ensures adding any two elements of G still gives an element in G.

Associativity: checks if (x + y) + z == x + (y + z) (mod 5).

Identity element: finds e such that x + e = x (mod 5). Here, e = 0.

Inverse elements: finds y for each x such that x + y = 0 (mod 5).
"""

# albelian group : a group is called albelian if the operation is commutative
# commutative means that a ⊗ b = b ⊗ a

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# VECTOR SPACES

# a vector space is a set of vectors that is closed under two operations : addition and scalar multiplication
# in my understanding  vector space allows vectors to be scaled, but the set itself is still closed under these operations
# like group was still confined to the same set but vector space can expand or contract the set
# a vector space is a set of vectors that is closed under two operations : addition and scalar multiplication
# think of vectors in 2D space:
# inner operation (+): adding two vectors → still a vector.
# outer operation (·): scaling a vector (stretching/shrinking it) → still a vector.

"""
money in a bank account (group vs vector space)
group perspective (only addition, no scaling):
imagine you have a bank account where you can only deposit or withdraw fixed amounts, say $10 at a time.

you start with $50, you add $10 → now you have $60.

you remove $10 → now you have $50 again.

you’re stuck using multiples of $10—you can’t scale the numbers, just add or subtract in fixed steps.

this is like a group: you have an operation (addition), but you're confined to the specific set of amounts you can reach.

vector space perspective (addition + scaling):
now imagine you can multiply your money—maybe by investing it or applying interest rates.

if you invest your $50 and it doubles, now you have $100.

if you lose half, you have $25.

you’re not just adding/subtracting fixed amounts anymore—you’re scaling your money up and down.

this is a vector space: you can still add money, but now you can also scale it up or down (like multiplying a vector by a scalar).

key difference:
groups = just adding/subtracting (fixed steps).

vector spaces = adding and stretching/shrinking (scaling).

"""

"""
A vector space is a collection of vectors that follow two main rules:

You can add vectors together → The result is still a vector.

You can scale (stretch/shrink) vectors using numbers (scalars) → The result is still a vector.

Operations in a Vector Space
Vector Addition (+):

Adding two vectors gives another vector in the same space.

Example: Moving two steps in one direction and three steps in another still keeps you in the same "space."

Scalar Multiplication (·):

Multiplying a vector by a number (scalar) stretches or shrinks it.

Example: Doubling a velocity vector makes you move twice as fast in the same direction.
"""

# Rules a vector space must follow:
# 1. (V, +) is an Abelian group:
#    - Adding two vectors keeps you in V
#    - Order doesn't matter: a + b = b + a
#    - There's a zero vector: x + 0 = x
#    - Every vector has an inverse: x + (-x) = 0
#
# 2. Distributivity:
#    - λ * (x + y) = λ * x + λ * y  # Scaling a sum is the same as scaling each part
#    - (λ + ψ) * x = λ * x + ψ * x  # Adding scalars before multiplying works the same way
#
# 3. Associativity of scaling:
#    - λ * (ψ * x) = (λψ) * x  # Scaling twice gives the same result
#
# 4. Identity for scaling:
#    - 1 * x = x  # Multiplying by 1 does nothing

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# VECTOR SUBSPACES  

# a vector subspace is a smaller set of vectors within a vector space  
# it behaves just like a vector space but is "contained" inside the bigger space  
# no matter how you add or scale vectors in the subspace, you never leave it  

# vector subspaces are smaller sets inside a vector space that stay closed under vector operations  
# basically, if you add two vectors in the subspace or scale them, you're still inside the subspace  

# for U to be a subspace of V:
# 1. it must have the zero vector (0 must be in U)
# 2. it must be closed under vector addition (x + y is still in U)
# 3. it must be closed under scalar multiplication (scaling a vector in U still gives a vector in U)

# example:  
# in 3D space, a flat plane passing through (0,0,0) can be a subspace  
# because adding two vectors in the plane keeps you in the plane  
# and scaling a vector still keeps you in the plane  
# The intersection of arbitrarily many subspaces is a subspace itself.

# but a random tilted surface or a plane not passing through the origin?  
# not a subspace, because scaling can take you outside it  

####### check animation and more details for it at animations\01_LA\vector_subspace.py


# every vector space V has two trivial subspaces:
# 1. the entire space V itself
# 2. the zero subspace {0} (just the origin)

# only some sets are valid subspaces:
# - for a set to be a subspace, it must be closed under vector addition and scalar multiplication.
# - if a set fails this, it is NOT a subspace.

# examples:
# - if a set does not include the zero vector, it is not a subspace.
# - if adding two vectors in the set gives a result outside the set, it is not a subspace.

# key properties:
# - the solution set of a **homogeneous system** (Ax = 0) is always a subspace.
# - the solution set of an **inhomogeneous system** (Ax = b, where b ≠ 0) is NOT a subspace.
# - the intersection of any number of subspaces is always a subspace.

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# LINEAR INDEPENDENCE

# vectors = elements of vector space

# first we will look into linear combination 

# linear combination notes

# a **linear combination** is when we scale and add vectors to create new vectors.

# given vectors x1, x2, ..., xk in a vector space V:
# any vector v that can be written as:
# v = λ1 * x1 + λ2 * x2 + ... + λk * xk
# (where λ1, λ2, ..., λk are scalars) is a linear combination.

# key properties:
# - the **zero vector** is always a linear combination:
#   0 = 0 * x1 + 0 * x2 + ... + 0 * xk
# - we focus on **non-trivial** linear combinations,
#   meaning at least one λi is **not zero** when representing 0.

# example:
# in R^2, vectors (1, 0) and (0, 1) can form any vector (a, b) by:
# (a, b) = a * (1, 0) + b * (0, 1)

# linear independence

# trivial linear combination : when all scalars are 0

# Linear Dependence & Independence

# Linear Combination:
# Any vector can be made by scaling & adding other vectors.

# Linearly Dependent:
# - At least one vector in the set can be made from the others.
# - Some information is redundant.
# - Example: (1,0), (0,1), and (2,2) → (2,2) is just 2*(1,1).

# Linearly Independent:
# - No vector in the set can be created using the others.
# - All vectors add something unique.
# - Example: (1,0) and (0,1) in 2D → Can't form one using the other.

# Shortcut: If removing a vector doesn’t change the "span," it was dependent.

# Properties of Linear Dependence & Independence

# 1. Vectors are either dependent or independent—no middle ground.
# 2. If at least one vector is zero or two vectors are identical → they are dependent.
# 3. If one vector is a multiple of another → they are dependent.
# 4. A set of k ≥ 2 vectors is dependent if at least one can be written as a combination of the others.

# Checking Independence: Gaussian Elimination
# - Write vectors as columns of a matrix A.
# - Perform Gaussian elimination to get row echelon form.
# - Pivot columns = independent vectors.
# - Non-pivot columns = dependent vectors (can be written as a combination of pivot columns).

# Shortcut:
# - If every column is a pivot column → vectors are independent.
# - If at least one column is not a pivot column → vectors are dependent.

# to remind  a pivot column is a column in a matrix that has the first nonzero entry (a leading 1 after row reduction) in a new row, indicating an independent vector.

# linear combinations

b1 = np.array([1,0])
b2 = np.array([0,1])

# scalars  
λ1, λ2 = 2, -3  

# linear combination: λ1 * b1 + λ2 * b2  
v = λ1 * b1 + λ2 * b2  
print(v)  # Output: [ 2 -3 ]

# 2. Checking Linear Independence
# Vectors are linearly independent if the only solution to Ax = 0 is x = 0.
# We check this using NumPy’s np.linalg.matrix_rank().
A = np.array([[1,2],[3,6]]) # two vectors as columns

rank = np.linalg.matrix_rank(A)

if rank == A.shape[1]: # number of columns
    print("linearly independent")
else:
    print("linearly dependent")

# rank of matrix is the number of linearly independent columns in the matrix

# Finding a Pivot Column

from sympy import Matrix  

A = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 0]])  # Third column is not a pivot
rref, pivot_columns = A.rref()
print("Pivot Columns:", pivot_columns)  # Output: (0, 1)

# 4. Checking if a Vector is in a Subspace
# To see if a vector 𝑣
# v belongs to a subspace spanned by some basis vectors, solve Ax = v.

A = np.array([[1, 2], [3, 4]])  # Basis vectors as columns
v = np.array([5, 11])  # Target vector

try:
    x = np.linalg.solve(A, v)  # Solve Ax = v
    print("v is in the subspace, solution:", x)
except np.linalg.LinAlgError:
    print("v is NOT in the subspace")  # If A is singular, no solution

# It checks if `v` is a linear combination of `A`'s columns by solving `Ax = v`; if successful, `v` is in the subspace.

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# BASIS and RANK

## generating set and span
"""
1. A generating set is a set of vectors {x1,x2....,xk} that can be used to form any vector in a vector space V using linear combinations
2. The span of {x1,x2,.....,xk} is the set of all possible linear combinations of the vectors
3. If the span of {x1,x2,...,x3} covers all of V we write :
                    V = span[x1,x2,...,xk]
4. In short a generating set defined a space and its span is the set of vectors that space can create.
"""

# basis

"""
A basis of a vector space V is a minimal generating set meaning:
    1. It spans V ( can form every vector in V using linear combinations )
    2. It is linearly independent ( no vector in the basis can be formed using the others )

No smaller set that the basis can still span V
"""

# Basis example in R²
# A basis is a set of linearly independent vectors that can represent any vector in the space.
# Let's define two basis vectors in R²:
b1 = np.array([2, 1])
b2 = np.array([-1, 1])

# Any vector in R² should be expressible as a linear combination of b1 and b2.
# Let's check if v = [3, 2] can be represented using b1 and b2.
A = np.column_stack((b1, b2))  # Form matrix with basis vectors as columns
v = np.array([3, 2])  # Target vector

# Solve Ax = v (i.e., find coefficients for the linear combination)
x = np.linalg.solve(A, v)

print("Coefficients for linear combination:", x)
# This means v = x[0] * b1 + x[1] * b2

# If A is invertible and the system has a unique solution, b1 and b2 form a basis.
# Solving Ax = v helps find the weights (coefficients) needed to express v using the basis.

# A basis is a minimal generating set and a maximal linearly independent set.

# 1. Minimal generating set:
#    - No smaller subset of B can still generate the entire space V.

# 2. Maximal linearly independent set:
#    - If we add any other vector to B, the set becomes linearly dependent.

# 3. Unique representation:
#    - Every vector x in V can be written uniquely as a linear combination of basis vectors.
#    - If two different sets of coefficients give the same vector, then those coefficients must be equal.

# This means that for a given basis B = {b1, b2, ..., bk}, 
# any vector x in V can be written uniquely as:
#    x = λ1 * b1 + λ2 * b2 + ... + λk * bk

# determining a basis using row echelon form
# ------------------------------------------
# 1. arrange vectors as columns of a matrix
# 2. perform row echelon reduction
# 3. identify pivot columns (columns with leading 1s in reduced form)
# 4. the pivot columns form a basis (they are linearly independent)

# example: vectors in R^3
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])  # matrix with 3 column vectors

# convert to sympy Matrix for row reduction
M = Matrix(A)
ref = M.rref()  # get row echelon form

print("Row Echelon Form:")
print(ref[0])  # reduced matrix

print("Pivot columns (basis indices):", ref[1])  # pivot column indices

# extract basis vectors from original matrix
basis_vectors = A[:, list(ref[1])]
print("Basis vectors:")
print(basis_vectors)

## rank of a matrix

# the number of linearly independent columns of a matrix equals the number of linearly independent rows and is called the rank of the given matrix

# some important properties of matrix
# ------------------------------
# MATRIX SPACES & RANK CONCEPTS
# ------------------------------

# The column space (span of columns) of A defines a subspace U in R^m
# The rank of A (rk(A)) gives the dimension of this subspace
# A basis for this subspace can be found by identifying pivot columns after row reduction

# The row space (span of rows) of A defines a subspace W in R^n
# The rank of A is also the dimension of this row space
# A basis for the row space can be found by applying row reduction to A^T (transpose of A)

# A square matrix A (n × n) is invertible (regular) if and only if rank(A) = n
# This means all columns are independent, and Ax = b always has a unique solution

# A system Ax = b has a solution if and only if rank(A) = rank([A|b])
# where [A|b] is the augmented matrix (A with b as an extra column)

# The null space (kernel) of A consists of all solutions to Ax = 0
# Its dimension is given by (n - rank(A)), meaning how many free variables exist in Ax = 0

# A matrix is full rank if its rank is the highest possible: rank(A) = min(m, n)
# If not full rank, it's rank deficient, meaning it has dependent columns or rows

# -----------------------
# NULL SPACE (KERNEL)
# -----------------------
# The null space of a matrix A consists of all vectors x such that Ax = 0.
# These are the vectors that A "annihilates" (maps to zero).
# It's useful for solving homogeneous systems and finding dependencies.

# -----------------------
# SINGULAR VALUE DECOMPOSITION (SVD)
# -----------------------
# SVD decomposes A into three matrices: U, Σ, and V^T.
# A = U Σ V^T
# - U: left singular vectors (columns span the column space)
# - Σ: diagonal matrix of singular values (rank information)
# - V^T: right singular vectors (rows span the row space and null space)
# The last columns of V^T span the null space.

# -----------------------
# QR DECOMPOSITION
# -----------------------
# QR decomposition factors a matrix A into:
# A = Q R
# - Q: an orthogonal matrix (basis for column space)  [An orthogonal matrix is a square matrix whose inverse is equal to its transpose, meaning multiplying the matrix by its transpose results in the identity matrix]
# - R: an upper triangular matrix (used for Gaussian elimination)
# The pivot columns (where R has nonzero diagonal entries) form a basis
# for the column space.
import scipy.linalg as la

# Define a matrix A
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

# Compute the rank of A
rank_A = np.linalg.matrix_rank(A)
print("Rank of A:", rank_A)  # Should be 2 (since rows/columns are dependent)

# -----------------------
# FINDING NULL SPACE
# -----------------------
# SVD decomposition gives us fundamental spaces of A
U, S, Vt = la.svd(A)

# The last (n - rank) columns of Vt form the null space basis
null_space_basis = Vt[rank_A:].T  # Transpose to get column vectors
print("Null space basis:\n", null_space_basis)

# -----------------------
# FINDING COLUMN SPACE (BASIS)
# -----------------------
# QR decomposition gives pivot columns, which form a basis for the column space
Q, R = np.linalg.qr(A)  # Q is orthonormal, R is upper triangular
pivot_columns = np.where(np.abs(R.diagonal()) > 1e-10)[0]  # Find pivot columns
print("Pivot columns (basis for column space):", pivot_columns)

# -----------------------
# FINDING ROW SPACE (BASIS)
# -----------------------
# Row space is the span of the independent rows in A
# This is equivalent to finding the column space of A^T
rank_AT = np.linalg.matrix_rank(A.T)
print("Rank of A^T (same as rank(A)):", rank_AT)

# The row space basis can be extracted from the R matrix in QR decomposition
print("Row space basis:\n", R[:rank_A])  # First rank_A rows of R

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# LINEAR MAPPINGS

# --- LINEAR MAPPINGS: PRESERVING VECTOR SPACE STRUCTURE ---
# A linear mapping (or transformation) is a function that maps vectors from one space (V) 
# to another (W) while preserving the fundamental vector space operations:
#   1. Addition: Φ(x + y) = Φ(x) + Φ(y)   (Preserves vector addition)
#   2. Scalar Multiplication: Φ(λx) = λΦ(x)   (Preserves scaling)
# 
# This means that a linear transformation does NOT distort the basic structure of vectors.
# It only "reshapes" or "repositions" them while keeping their relationships intact.
# 
# Example: Rotations, scaling, reflections, and projections are all linear mappings.

A = np.array([[2,0],
              [0,1]]) # a 2x2 transformation matrix

# defining some vectors in 2D space

v1 = np.array([1,2])
v2 = np.array([3,4])

# --- check addition property ----

lhs = A @ (v1 + v2) # applying transformation to a sum
rhs = A@v1 + A@v2 # transforming individually and adding

print("checking addition property : ",np.allclose(lhs,rhs)) # should be true 
# what np.allclose basically does is it checks if the two arrays are equal within some tolerance aka it checks if the two arrays are almost equal

# chcek scalar multiplication property
scalar = 5
lhs = A @ (scalar * v1)
rhs = scalar * (A@v1)

print("Scalar multiplication property status : ", np.allclose(lhs,rhs)) # should be true


# A mapping (or function) Φ: V → W describes how each element of V (domain) maps to W (codomain).
# The properties of this mapping define its structure.

# Injective (One-to-One) Mapping:
# - Each input (x) maps to a unique output (Φ(x)).
# - No two different inputs produce the same output.
# - Mathematically: Φ(x) = Φ(y)  →  x = y (i.e., if outputs are equal, inputs must have been equal).

# Surjective (Onto) Mapping:
# - Every element of W (codomain) is covered by Φ.
# - There are no "unreachable" elements in W.
# - Mathematically: ∀ w ∈ W, ∃ x ∈ V such that Φ(x) = w (every element in W is mapped to by some x).

# Bijective Mapping:
# - A function that is both injective and surjective.
# - This means:
#   1. Each output in W is uniquely mapped from V.
#   2. Every element of W is reached.
# - Bijective functions have an inverse Φ⁻¹ such that applying it brings us back to the original input.

# Special Types of Linear Mappings in Vector Spaces:
# - Isomorphism: A linear and bijective map between two vector spaces (preserves structure).
# - Endomorphism: A linear map from V to itself (Φ: V → V).
# - Automorphism: A bijective linear map from V to itself (invertible endomorphism).
# - Identity Mapping: The function idV(x) = x, which leaves every element unchanged.


# Example transformation matrix
A = np.array([[1, 2], 
              [3, 4]])

# Injectivity Check: If A is full column rank (i.e., rank(A) = number of columns), it's injective
is_injective = np.linalg.matrix_rank(A) == A.shape[1]
print("Injective:", is_injective)  # True if no two inputs map to the same output

# Surjectivity Check: If A is full row rank (i.e., rank(A) = number of rows), it's surjective
is_surjective = np.linalg.matrix_rank(A) == A.shape[0]
print("Surjective:", is_surjective)  # True if every element in W can be reached

# Bijectivity Check: A must be square (n×n) and full rank
is_bijective = A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]
print("Bijective:", is_bijective)  # True if invertible (both injective & surjective)

# If bijective, find inverse
if is_bijective:
    A_inv = np.linalg.inv(A)
    print("Inverse of A:\n", A_inv)  # Exists only if A is bijective

# We check injectivity using rank because a transformation is injective if its columns are linearly independent (i.e., rank = number of columns), and we check surjectivity because a transformation is surjective if its rows span the entire codomain (i.e., rank = number of rows).

# If a transformation is bijective, it has an inverse that can "undo" the mapping.

# Theorem: Two finite-dimensional vector spaces V and W are isomorphic 
# (structurally identical) if and only if they have the same dimension.
# 
# Intuition:
# - If dim(V) = dim(W), there exists a bijective linear map between them.
# - This means they can be transformed into each other without loss of information.
# - Example: The space of m × n matrices (R^(m×n)) and the space of vectors of length mn (R^(mn)) 
#   are essentially the same since both have dimension mn.
#
# Important consequences:
# 1. Function composition preserves linearity:
#    If Φ : V → W and Ψ : W → X are linear, then Ψ ◦ Φ : V → X is also linear.
# 2. Isomorphisms are invertible:
#    If Φ : V → W is an isomorphism, its inverse Φ⁻¹ : W → V is also an isomorphism.


# Example: Transforming a matrix space R^(m×n) to vector space R^(mn)
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2 matrix in R^(3x2)
vectorized_A = A.flatten()  # Convert it to a 6D vector in R^6
reshaped_A = vectorized_A.reshape(3, 2)  # Convert back to original matrix

print("Original matrix:\n", A)
print("Vectorized form (in R^6):", vectorized_A)
print("Reshaped back to matrix:\n", reshaped_A)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# Matrix Representation of Linear Mappings

import numpy as np

# Define a new basis (b1, b2) as column vectors of a matrix B
B = np.array([[1, 2],  # First basis vector b1
              [3, 4]])  # Second basis vector b2

# Define a vector x in standard Cartesian coordinates (relative to e1, e2)
x_standard = np.array([2, 2])  # x = 2*e1 + 2*e2

# To express x in terms of basis B, we solve B * alpha = x for alpha
# alpha represents the coordinates of x in the new basis B
alpha = np.linalg.solve(B, x_standard)

print("New coordinates of x in basis B:", alpha)

# Convert back: reconstruct x from alpha using the basis B
x_reconstructed = B @ alpha
print("Reconstructed x (should match original x):", x_reconstructed)

# Transformation Matrix: A linear mapping Φ: V → W can be represented by a matrix A
# where the columns are the images of the basis vectors in W (Φ(b1), Φ(b2), ...).
# To transform a vector x in V to y in W, we compute y = A * x.

# Define the transformation matrix AΦ (maps from V to W)
A_Phi = np.array([[2, 1],  # First column: Φ(b1) in C basis
                  [0, 3]])  # Second column: Φ(b2) in C basis

# Define a vector x in V with respect to basis B
x_B = np.array([4, 5])  # Coordinates of x in basis B

# Compute transformed vector y in W with respect to basis C
y_C = A_Phi @ x_B  # Matrix-vector multiplication: y = A_Phi * x
print("Transformed vector y in basis C:", y_C)

# Verification: If given y in W, we can retrieve x in V (if A_Phi is invertible)
if np.linalg.det(A_Phi) != 0:  # Ensure the matrix is invertible
    A_Phi_inv = np.linalg.inv(A_Phi)  # Compute inverse transformation
    x_recovered = A_Phi_inv @ y_C
    print("Recovered x in basis B:", x_recovered)

# --------------------------
# BASIS CHANGE

# --------------------------------------------
# Basis Change & Transformation Matrices
# --------------------------------------------

# 1. Consider a linear mapping Φ: V → W.
#    The transformation matrix A_Phi represents this mapping in some bases B (for V) and C (for W).

A_Phi = np.array([[2, 1],  # Transformation matrix in basis (B, C)
                  [0, 3]])  # Maps V -> W

# 2. Suppose we change the basis in V from B to B̃.
#    Similarly, we change the basis in W from C to C̃.
#    The matrices P_B and P_C represent these basis changes.

P_B = np.array([[1, 2],  # Change of basis matrix for V (B → B̃)
                [3, 4]])

P_C = np.array([[0, 1],  # Change of basis matrix for W (C → C̃)
                [1, 0]])

# 3. To correctly represent the same transformation in the new bases,
#    we need to compute the updated transformation matrix A_Phi_tilde.
#    The formula for this is:
#
#       Ã_Phi = P_C⁻¹ * A_Phi * P_B
#
#    where:
#       - P_B changes coordinates from B̃ to B
#       - P_C changes coordinates from C̃ to C
#       - P_C⁻¹ reverts the change in W

P_C_inv = np.linalg.inv(P_C)  # Compute inverse of P_C

A_Phi_tilde = P_C_inv @ A_Phi @ P_B  # Apply basis change formula

# 4. The result is the transformation matrix Ã_Phi,
#    which represents the same mapping Φ in the new bases (B̃, C̃).

print("Original transformation matrix A_Phi (in basis B, C):\n", A_Phi)
print("\nNew transformation matrix Ã_Phi (in basis B̃, C̃):\n", A_Phi_tilde)

# in simple terms : It’s about how linear transformations change when you switch to a different coordinate system (basis) in vector spaces.

# Think of a basis like a coordinate system or measurement standard.
# Example: In a city, you can describe locations using:
# - A street grid: "2 blocks north, 3 blocks east"
# - Landmarks: "Halfway between the library and the mall"
# Both describe the same point but in different coordinate systems (bases).

# Now, consider a transformation (like a bus route moving people):
# - In the street grid: "Move 3 blocks east, then 2 blocks south"
# - In the landmark system: "Move away from the mall towards the stadium"
# The movement stays the same, but its description changes.

# In linear algebra, when we change the basis, the transformation matrix 
# (which represents the movement) also changes. The transformation itself 
# remains the same, but its matrix form depends on the chosen basis.

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# (Basis Change)
# When we change the basis in vector spaces V and W, the transformation matrix A_Φ 
# also changes accordingly. The new transformation matrix ˜A_Φ is computed as:
# 
# ˜A_Φ = T⁻¹ * A_Φ * S
#
# - S is the transformation matrix that converts coordinates from the new basis ˜B to the old basis B in V.
# - T is the transformation matrix that converts coordinates from the new basis ˜C to the old basis C in W.
# - The new transformation matrix ˜A_Φ adjusts A_Φ so that it works correctly with the new bases.
# 
# Intuition:
# Imagine changing the way we describe points in space (e.g., switching from 
# Cartesian to rotated axes). The transformation rules must also adjust accordingly 
# to maintain the same mapping but in the new coordinate system.


# Original transformation matrix A_Φ in the standard basis
A_phi = np.array([[2, 1], 
                  [1, 2]])

# Change of basis matrices  
# S transforms new basis coordinates to old basis coordinates in V
S = np.array([[1, 1], 
              [1, -1]])  # Columns represent new basis vectors in terms of old basis

# T transforms new basis coordinates to old basis coordinates in W
T = np.array([[1, 0], 
              [0, 1]])  # Identity, meaning same basis in W for simplicity

# Compute the new transformation matrix in the new basis
S_inv = np.linalg.inv(S)   # Compute inverse of S
T_inv = np.linalg.inv(T)   # Compute inverse of T (which is just identity here)

A_phi_new = T_inv @ A_phi @ S  # Apply basis change formula: ˜A_Φ = T⁻¹ * A_Φ * S

print("Original transformation matrix A_Φ:")
print(A_phi)

print("\nNew transformation matrix ˜A_Φ after basis change:")
print(A_phi_new)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

#  (Equivalence)
# Two matrices A and ˜A (both m x n) are equivalent if there exist
# two invertible matrices S (n x n) and T (m x m) such that:
# ˜A = T⁻¹ A S
# This means A and ˜A represent the same linear transformation under different bases.

#  (Similarity)
# Two square matrices A and ˜A (both n x n) are similar if there exists
# an invertible matrix S (n x n) such that:
# ˜A = S⁻¹ A S
# This means A and ˜A describe the same transformation but in different coordinate systems.

# Remark:
# - Similar matrices are always equivalent.
# - Equivalent matrices are not necessarily similar.

# Basis change as function composition:
# - AΦ represents a transformation from V to W using basis B and C.
# - ˜AΦ represents the same transformation but using new bases ˜B and ˜C.
# - S changes how vectors in V are represented (old basis B → new basis ˜B).
# - T changes how vectors in W are represented (old basis C → new basis ˜C).

# The new transformation is found by:
# ˜AΦ = T⁻¹ AΦ S
# This is like stacking functions:
# (change W basis) ∘ (original transformation) ∘ (change V basis)

"""
%%%%%%%%######%%%%%@
##%%%#**+==+==*%%%%%
##%@##**++=++++#%%%%
##%%####++++#++*%%%%
++*##*++++++===+#%%%
+==+*##***+++++#%%%%
*****####**+++*+*%%%
****#####******===+#
++***#########++++++
++****#######***++++
++****########******
"""

# ---------------------------
# Image and Kernel
# --------------------------

# day 1 done here , will continue from tommorrow hopefully 

























































"""

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%#%%%%%%@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@
%%%%####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%%%%%%%%%%%#%%%#*#%##***#%%%%#**#***%%%%**##**++*##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@
%#####%%%%##%###%%%%%%%%%%%%%%%%%%%%%%%%%#******+++*%%****+++++===+++=====+++#%#===========+#%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%@@@@@@@@@@
#####%%%######%%%%%%%%%%%%%%%%%%%%%%%%%%#******++++###*+=+=================++*%#+===========++*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%@@@@@@@@
####%%#######%%%%%%%%%%%%%%%%%%%%%%%%%%#******++++*##*++=+==================+*##*============++*#*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%@@
###%########%%%%##%%%%%@@@%%@%%%%%###%#******+++++*##*+++===================+###*============++*###=*##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%#####**#%%%###%%%%@@@@%%%%%%@%%%#%%#*****++++++##**+++===================+*##*============+++#%+==+%%%%%%%%%%%%%%%%%%%%%%%@%%%%%%%%%%
#######**#%%#%###%%@@@@@%%%%%%@%%#**%%#****+++++++##**+++==================+++*#*============++*##+===*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###***#%%%%#%###%@@@@@@@%%@%@@%%%#*#%%#****+++++++##*++++==================+=+*#*============+*+##+=+*+*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#######%%%%%%%%@@@@@@@@@%@@@@@%%#**#%##****+++++++*#*++++====================+*#*===========+++=##+==*#%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%
#######%#%#####%@@@@@@@@%@@@@@%##**#%#*#**++++++++*##++++++==================+*#*============++=##====#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
######%######%%@@@@@@@@@%@@@@@%##**#%#**#*+++++++++##++++++==================+##*============+==#*=====#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
######%####%%%%@@@@@@@@@%@@@@%%%***#%#****+++++++++*#*+++++==================+##+============+=+#+===+#*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####%##%%%%%#%%@@@@@@@@@@@@@%#*****##****++++++++++#*++++++=================*#*===============**=====+%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##############%%%#@@@@@@@@@@%%#*****##****+++++++++*#%#*+++++======++=======+*#================*=======*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##*##########%%%*%%@@@@@@@@@%%#****#%%%%%%%%%%%%%%%%%%%%%##*+++++===++=====+#%%######****#######*=====+#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
****#########%%#*%%@@@@@@@@@%%#*##################****##**##*++++=====++*###%%%#%##########*++++***+==*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##**###*#####%%**#%@@@@@@@@%%###******#****************##******++=======+++#*********+++==============+#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##*********##%#**#%@@@@@@@@@%#********##*********#******#*****+++========+*****++*#####*+++++==========*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##*********##%***%%@@@@@@@@%%#*********############****##***+++++========++**+=+#*+++*#%##*#**+++=======#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##**********#%**#%@@@@@@@@@%%#**********#%***#%%%%%%*++*#***++++++=+==========+**==+#####%#+++#*========+%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##**********###%%%%@@@@%%%%%##*********#%#*+*%%%%%%%%+==*++++++++++============*===*##%%%##+==+#+=======+#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##***#****##%%#***%%%%%##%%%###*********#*+++%#######+==++++++++++++++=============**+***##=--=**=======+%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##**********##***#%%%%#####%%#********####++++##*###+==+++++++++++++++++====+++=====+****+=--=++========+%%%+==*%%%%%%%%%%%%%%%%%%%%%%%%
+****************#*#%######%%#####******#**+++++++++++++*+++++++++++++++======+++++======--=============*%%+=+*=*%%%%%%%%%%%%%%%%%%%%%%%
*+************+++#**%%######%%##**********+++++++++++++++++++++++++++++++===========+===================##*=*+*=*%%%%%%%%%%%%%%%%%%%%%%%
++***********#++**++#%#######%%###*******++++++++++++++++++++++++++++++++++++==========================+##**++++#%%%%%%%%%%%%%%%%%%%%%%%
++++++++++++***++*++#%%#######%###******++++++++++++++++++++++++++++++++++++===========================*#**+++=*%%%%%%%%%%%%%%%%%%%%%%%%
=-=========+*****#*#%%#########%####*****++++++++++++++++++++++++++++++++++++++========================##**+++=#%%%%%%%%%%%%%%%%%%%%%%%%
=============++*****##**#######%#####*****++++++++++++++++++++++*+++++++++++++++======================*#*++*+=*%%%%%%%%%%%%%%%%%%%%%%%%%
===================*+===+#######%######*****+++++++++++++++++****++++++++++++++++++++=================#+****++#%%%%%%%%%%%%%%%%%%%%%%%%%
=========================*#######%#####*******++++++++++++*******++++++++++++++++++++++==============**+***++#%%%%%%%%%%%%%%%%%%%%%%%%%%
==========================*##############*********++++***********++++++++++++++++++++++++===========+*****+*%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=====++====================+*##############***********************+++++++++++++++++++++++===========****++#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
++++++++++++=====+===========+*#############**#*******************+++++++++++++++++++++++++++======*++=+#%%%%%%%%%%%%%%%%%%%%%%%%#**#%%%
**+++++++*+++++++=+++==========+*###############******************+++++++++++++++++++++++++++++=++++++*#%%%%%%%%%%%%%%%%%%%%%%%%###%%%%%
***+++++++*++++++++++++=========+*################****************+++++++++++++++++++++++++++++++++++*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
****+++++*****++++++++++========+++################***************+++++++++++++++++++++++++++++++#%#+==+#%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%
******+****************+=======+***###%%#############**************++**+++++++++++++++++++++++++#%#*++===+#%%%%%%%%%%%%%%%%%%%%%%#%%%%%%
**************#********++====+*******#%%%##############*********#*****#**++++++++++++++++++++++#%%*++=+=====*#%%%%%%%%%%%%%%%%%%##%%%%%%
*******#******#*********+=+*********##%%%%##############*****************+++++++++++++++++++++#%%%*++=+=======+#%%%%%%%%%%%%%%%%#%%%%@@@
##*****####*###****####*********#######%%%%###############*************++++++++++++++++++++++%%%*#*++==+========+#%%%%%%%%%%%%%##%%%%@@@
#######***###*+++*####********##########%%%%################*#********+++++++++++++++++++++*#%%#****+===+==========*#%%%%%%%%%%#%%%%@@@@
#######****#******#**********############%%%%##################******+++++++++++++++++++++*%#%#*****+===+============+#%%%%%%%##%%%%@@@@
***********#****************##############%##%##################**********+++++++++++++++*####******+====+==============*%%%%%#%%%%%%@@@
**********###*************####################%########################***#*+++++++++++*##**#*******+=====+===============+####%%%%%%@@@
*************+************#####################%%#############*********+++**+++++++++*******#******++=====+==================+*#%%%%%%%%
+********+++++************######################%%###############*************++++++*#**#**********++=====++====================+*#%%%%%
*******+++++++***********########################%###############****************+**#**#***********++++++==++======================+*%%%
****++++++++++************###########################################*************##**##*****#*****+++++++++++========================+#
**+++++++++++**************########################################*************###**###*****#*****++++++++++++++=+=====================
+++++++++++++***************#####################################*************##########*****#****++++++++++++++++++++++================
++++++++++++++***************####################################**********################**#***++++++++++++++++++++++++++++++++=======
+++++++++++++***************#*#############################################################*##**++++++++++++++*+++++++++++++++++++++++++
+++++++++++++*******************#############%################################################*++++++++++++++++*++++++++++++++++++++++++
++++++++++++++*******************###########################################################***+++++++++++++++++++++++++++++++++++++++++
++++++++++++++*********************########################################################****+++++++++++++++++++++++++++++++++++++++++
+++++++++++++***********************######################################################******++++++++++++++++*+++++++++++++++++++++++
+++++++++++++*************************######%%##########################################********++++++++++++++++**++++++++++++++++++++++
+++++++++++++************#**************####%%#############%##########################************+++++++++++++++*++++++++++++++++++++++
++++++++++++++****************************####%#############%########################*********#******+++++++++++++*+++++++++++++++++++++
++++++++++++++******************************###%###################################**********#***********+++++++++**++++++++++++++++++++
++++++++++++++******************************#####%###########%####################**********#********************+**++++++++++++++++++++
+++++++++++++*******************************#######%#########%##################*********####***************************++++++++++++++++
++++++++++++++******************************#######################################*########**#*******************************+*++++++++
++++++++++++++******************************##########################################**#####*#*****************************************
+++++++++**+++********#********************####################################################*****************************************
++++++++++**++*****************************####################################################*****************************************
=++++++++++******************************######################################################*****************************************
-=++++++++++****************************#######################################################*****************************************
---++++++++++************#************#########################################################****************#***********************#

"""