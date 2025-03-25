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