################################################
############# ANALYTIC GEOMETRY ################
################################################

# my notes with examples and code for analytic geometry , requires knowledge of 01_LA.py

# 1. Norms
import numpy as np

# ───────────────────────────────────────────────
# 💡 NORM = generalized length of a vector
# ∥x∥ is a scalar ≥ 0 that tells you how "big" or "long" a vector x is
# ───────────────────────────────────────────────

# Example vector
x = np.array([3, 4])  # classic 3-4-5 triangle vector

# ℓ2 norm (Euclidean norm) = sqrt(x1² + x2² + ... + xn²)
# This is the "default" norm unless stated otherwise
euclidean_norm = np.linalg.norm(x)
print("∥x∥₂ =", euclidean_norm)  # output: 5.0

# ───────────────────────────────────────────────
# 🔁 PROPERTY 1: Absolute Homogeneity
# ∥λx∥ = |λ| * ∥x∥
# ───────────────────────────────────────────────
λ = -2
lhs = np.linalg.norm(λ * x)
rhs = abs(λ) * np.linalg.norm(x)
print("Homogeneity holds:", np.isclose(lhs, rhs))  # should be True

# ───────────────────────────────────────────────
# 🔁 PROPERTY 2: Triangle Inequality
# ∥x + y∥ ≤ ∥x∥ + ∥y∥
# ───────────────────────────────────────────────
y = np.array([1, -1])
left = np.linalg.norm(x + y)
right = np.linalg.norm(x) + np.linalg.norm(y)
print("Triangle inequality holds:", left <= right)  # should be True

# ───────────────────────────────────────────────
# 🔁 PROPERTY 3: Positive Definiteness
# ∥x∥ ≥ 0, and ∥x∥ = 0 ⇔ x = 0
# ───────────────────────────────────────────────
zero_vector = np.array([0, 0, 0])
norm_zero = np.linalg.norm(zero_vector)
print("Positive definiteness holds:", norm_zero == 0)  # True only if vector is zero

# ───────────────────────────────────────────────
# 🔄 BONUS: Other norms
# ───────────────────────────────────────────────

# ℓ1 norm = Manhattan distance = sum of absolute values
l1_norm = np.linalg.norm(x, ord=1)  # ∥x∥₁ = |3| + |4| = 7
print("∥x∥₁ =", l1_norm)

# ℓ∞ norm = max absolute value in vector
linf_norm = np.linalg.norm(x, ord=np.inf)  # max(|3|, |4|) = 4
print("∥x∥∞ =", linf_norm)

# in simple words
"""
A norm is a function that tells us how long a vector is - consider it to be a measuring tape in any vector space

A norm ∥·∥ is a function:

 ∥·∥ : V → ℝ
x ↦ ∥x∥ 

It must satisfy 3 properties for all scalars λ and all vectors x, y in V:

1. Absolute homogeneity : ∥λx∥ = |λ| ∥x∥
    scaling a vector scales its norm by the absolute value of the scalar.

2. Triangle Inequality : ∥x + y∥ ≤ ∥x∥ + ∥y∥
    the norm of a sum is less than or equal to the sum of the norms , or in simple words , going from A to B to C is never shorter than directly from A to C.

3. Positive definiteness : ∥x∥ ≥ 0, and ∥x∥ = 0 ⇔ x = 0
    
        the norm is always non-negative and is zero if and only if the vector itself is the zero vector , or in simple words , length is always non-negative, and only zero when the vector is zero.

"""

"""
::::--=+=+*###++++**
.:-=++****+*#%#+===#
-==***#%%%%%%##**++#
=+++*%@%%%%%@@%#***#
=:-+#@@%%%%%%%@*++=+
=-:-#%*#*++**=%%+---
*+=+#*---:.:--+*----
%#*=**#=....=#+----=
@@@%*-*=+**+=*--===+
@@@@#=*#%%@%##=+###%
##%%%%%%%%%%%%%%####
"""
import numpy as np

# ─────────────────────────────────────────────
# 📌 EXAMPLE 3.1 - MANHATTAN NORM (ℓ1 norm)
# ∥x∥₁ = sum of absolute values of the vector components
# Think: total blocks you'd walk in a grid-like city like NYC
# ─────────────────────────────────────────────

x1 = np.array([3, -4])  # sample vector
manhattan_norm = np.sum(np.abs(x1))  # manual calc
print("∥x∥₁ (Manhattan) =", manhattan_norm)  # → 7

# using numpy's built-in version:
l1_np = np.linalg.norm(x1, ord=1)
print("∥x∥₁ via np =", l1_np)

# ─────────────────────────────────────────────
# 📌 EXAMPLE 3.2 - EUCLIDEAN NORM (ℓ2 norm)
# ∥x∥₂ = sqrt(sum of squares) = usual "distance from origin"
# Think: crow-flies distance in 2D/3D space
# ─────────────────────────────────────────────

x2 = np.array([3, 4])
euclidean_norm = np.sqrt(np.sum(x2**2))  # manual
print("∥x∥₂ (Euclidean) =", euclidean_norm)  # → 5

# using numpy:
l2_np = np.linalg.norm(x2)  # default is ord=2
print("∥x∥₂ via np =", l2_np)

# ─────────────────────────────────────────────
# 💡 Summary:
# ℓ1 norm = ∑|xi| → grid walk distance
# ℓ2 norm = sqrt(xᵀx) → Pythagorean distance
# both are valid "lengths" in vector space, used in diff contexts
# ─────────────────────────────────────────────

# animation for this is at 02_AG/euc_man.py

"""
::::--=+=+*###++++**
.:-=++****+*#%#+===#
-==***#%%%%%%##**++#
=+++*%@%%%%%@@%#***#
=:-+#@@%%%%%%%@*++=+
=-:-#%*#*++**=%%+---
*+=+#*---:.:--+*----
%#*=**#=....=#+----=
@@@%*-*=+**+=*--===+
@@@@#=*#%%@%##=+###%
##%%%%%%%%%%%%%%####
"""

# heads up : throughout the book it is suggested to use euclidean and hence i will be following the same in the script unless mentioned otherwise

####################
# INNER PRODUCTS
####################

# 1. Dot Product

# inner products give us a way to talk about:
# - length (norms)
# - angles
# - projections
# - orthogonality (whether vectors are perpendicular)

# the most common inner product in R^n is the dot product:
#    x ⋅ y = sum(x_i * y_i)
# this gives us a scalar (just a number).

# if dot(x, y) == 0 → x and y are orthogonal (perpendicular).
# if dot(x, y) > 0 → acute angle
# if dot(x, y) < 0 → obtuse angle

# dot product is also useful for:
# - projecting vectors
# - cosine similarity
# - checking alignment between directions

# define two vectors
x = np.array([1, 2])
y = np.array([3, 4])

# dot product using numpy
dot = np.dot(x, y)
print(f"dot(x, y): {dot}")  # should be 1*3 + 2*4 = 11

# norm (length) of vectors
norm_x = np.linalg.norm(x)
norm_y = np.linalg.norm(y)

# angle between x and y using cosine formula
cos_theta = dot / (norm_x * norm_y)
angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)

print(f"angle between x and y (deg): {angle_deg:.2f}")

# check orthogonality
if np.isclose(dot, 0):
    print("vectors are orthogonal")
else:
    print("vectors are NOT orthogonal")


# General Inner Products

# an inner product is just a rule that:
# - takes in 2 vectors
# - spits out a number
# - behaves nicely with addition, scalar multiplication, symmetry, and positivity

# if it follows:
# 1. linear in both inputs (bilinear)
# 2. symmetric → ⟨x, y⟩ = ⟨y, x⟩
# 3. positive definite → ⟨x, x⟩ > 0 for x ≠ 0
# then congrats, it's an inner product space

# create a symmetric, positive-definite matrix A
A = np.array([[2, 0],
              [0, 3]])

# define a custom inner product using A: ⟨x, y⟩ = xᵀAy
def inner_product(x, y):
    return np.dot(x, A @ y)

# test vectors
x = np.array([1, 2])
y = np.array([3, 4])

# use our custom inner product
print(f"⟨x, y⟩ = {inner_product(x, y)}")

# check symmetry
print(f"⟨x, y⟩ == ⟨y, x⟩? {inner_product(x, y) == inner_product(y, x)}")

# check positive definiteness
print(f"⟨x, x⟩ = {inner_product(x, x)}")  # should be > 0

# in simple words
# the thing is that dot product only outputs a numebr but general inner product is an upgrade , its a custom rule capable of measuring:
# length
# angles 
# orthogonality
# but in more expanded space not restricted to R^n

# whats needed to make it an inner product

"""
your custom rule (aka function) must:

1 . be bilinear → linear in both inputs

2. be symmetric → order doesn’t matter (⟨x, y⟩ = ⟨y, x⟩)

3. be positive definite → ⟨x, x⟩ is always > 0 unless x is zero
"""

# Example - Inner Product That Is Not the Dot Product
# Define a custom inner product as per the given formula:
# ⟨x, y⟩ := x1y1 − (x1y2 + x2y1) + 2x2y2

def custom_inner_product(x, y):
    return x[0] * y[0] - (x[0] * y[1] + x[1] * y[0]) + 2 * x[1] * y[1]

# Test vectors
x = np.array([1, 2])
y = np.array([3, 4])

# Compute the custom inner product
result = custom_inner_product(x, y)
print(f"⟨x, y⟩ (custom) = {result}")

# Verify symmetry: ⟨x, y⟩ == ⟨y, x⟩
symmetric = custom_inner_product(x, y) == custom_inner_product(y, x)
print(f"Symmetry holds? {symmetric}")

# Verify positive definiteness: ⟨x, x⟩ > 0 for x ≠ 0
positive_definite = custom_inner_product(x, x) > 0
print(f"Positive definiteness holds? {positive_definite}")

# animation at 02_AG/in_po.py

"""
::::--=+=+*###++++**
.:-=++****+*#%#+===#
-==***#%%%%%%##**++#
=+++*%@%%%%%@@%#***#
=:-+#@@%%%%%%%@*++=+
=-:-#%*#*++**=%%+---
*+=+#*---:.:--+*----
%#*=**#=....=#+----=
@@@%*-*=+**+=*--===+
@@@@#=*#%%@%##=+###%
##%%%%%%%%%%%%%%####
"""




















"""

------++=-----------::---::::--:::::::::::::::---+++++**#*#######*=++++++++====++===+++++===+++++++=
:::::-------::::::::-----------::+*=--:::::::::-**+*++**#*########****++++==+++++++=++*+++++++++++++
:::::::::::::::::::----------:::=*##***-:::::**###*****#**####%#####***+++++++***+++*************+**
:::::::::::::::::::---------:::-**##***#*::+##**#*#**+****###########**+++********+*******#######**#
................:---------:::::+**#****###*#**#****#*****##**########*+*******++++=++===+*+++###***#
................:--==---::::::=*******###%%###****##*****##########%#********+=======--------=*##*#*
.............:::-++##***-::::-*****#*####%%########*****###########%%#*******+=======--------==*####
.........:::----+++*#*++#+.:##*******##############****###########%%%%%###**++++======-------==*####
..:....:---=====+++*#***####*#*****#########*=-::::-=*############%%%%%%%#**++++++=======---===+####
.::::::--======++++**##*######********+:..:-=+*####*+=-::.:=*#####%%%%%%#*******+++====----====+####
:::::--=====---+++******#####*****+-::+#%%@@@@@@@@@@@@@@%%%*-::+##%%%%%%#*******+++++==---====+#####
:----==+=--:::+++***#***##*##***-:-#%%@@@@@@@@@@@@@@@@@@@@@%%%%*::*%%%%%%#******+++++++=--===+*#####
--===+=----.:-*+*****####*###+::*%@@@@@@%%%%%%%%%%%%%%%%%%%%@@@@%%=:+%%%%%#*******++++++=====+*#####
-==++=====-+#***+******##***::#%@@@@%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@%+:#%%%%##*********++=====+######
======++==****++++**+**#**=:#%@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@%-=%%%%%#*********+++==+*######
==++++++*#***++++++******=-%@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@+=%%%%#**********+++***######
=+++++++****++++++******==%@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@**%%%##*#####***+=+****#####
==++++++**++++++++*****+-%@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@=######***###**+=+******###
--==+++=+++++++++******-%@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@%+###*****###**+=+*****####
---:-=====+++++++******+@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@*###*+++*###*+=++*********
::..:-----=+++++++****=%@@@@@@@@@%%%%%%%%%##%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@%**+++++*###*+=++++******+
--::-::::-=====+++***+=@@@@@@@@@@@@%%%%%%%#####%%%%%%%%%%%%@@@@@@@@@@@@@@@@+++++++*##*===+++++++++++
===-:...::-::-==+++++++@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@%%@@@#++++++*#*+==========++++
+++=-::.:::::-===++++++@@@@@@@@@@@@@@@@@@@@%%%%%%%%%##****###**+%+***%@@@@+**===++**+===============
++++=--::::::---==+++=+@@@@@@@@@@@@@@@@%%%##******+++++#+*+++++**++**#@@@@@@%*==+*#*===============-
==++==--:::::::--=====*@@@@@@@@@@@@####****+++++++++++++++*++++++++++*%@@@@@@%*+**+=======---------:
===++==-:::::::--=====+@@@@@@%%#*++**+++%*++++++======+++++#*+++*+++++#%%@@@@@%***======-----:::--::
---++==--::::::--=====*%%@@@@%**+++***%*****+=========++*****#*+*-=+=-*#%@@@@@@#+======-----::-====:
--+++==---:::::---==-=@@@@@@%#+*++=%%@@%%%@%++========++@@%%@@@##:-:::*##%@@@@@%======------:::::-::
****++==--::::----+==%@@@@%%#*+++=*@@**%%%%+%+=========**%%%%*+@@+:..:*%##@@@@@@---=--------:::::=::
#****++=---::---=-=-%@@@@@###*=-::++#+=*==+==============+-=*=+*+::::-+++***#*@*-----------:-::::+::
#*****++++=-----=--=%@*#%####*=-::-+==++++====-:.......:-==+++-:=::::+*******@@:-----==-------:::=-:
#*****++==----=-=--+@@@@#******=-::=:-----::.....::........:::::-+::=+**++#@@@::--==--=-----:--::+::
******+++-:-=----=-+@@@@@@@*--+*--==::..........::.............-+--=*-:+*#@@@=::--=---=-------=::+:-
%%%%#*+=++==%*-----=%@@@@@@***#+*--#=:........................:=#-=#####+#%@%..:-:----=---=---==:+-+
%%%##*+=**%%@@##@@%##@@@@@@*****+##+=::.......................-==%%+*##*+#.+=...--:------------+-++-
%%%%#*+==*%%@@%=:::--+@@@@%******###*-:.......................-*%#****#+=*.#*.::-::--:--:-:---:-**-:
%%%%@%#%%##%@+::--:=-:%@-:#++***%###=:=......................--=%#*%#*#==+.@=:::-::--:--:-:-----**-:
%%%%@%*%%#-=*=#*---=-:#=::*+-++*%###=..=:..................:+..=#**%#**==+.:::::-::-------:----:++::
@%%@@%##%%+%#+*#%#**=:%=::+=-=+*%#*#=....-=:.........:...-=....=#**%#*+===::::::=::-------------*=:-
@@@@@@@@@@@@@@@@@@@@%:%+::=-:-+*##*#=......##=::......:#%:.....=#**%#*-==-:::---=---------------*=--
@@@@@@@@@@@@@@@@@@@@#:%*=-::-:=#%#*#=......*###%*=-+%##**:...:.+%**%%+-=--::----=--==--====+====*=-=
@@@@@@@@@@@@@@@@@@@@@--=**-:-:=*#**#=--:+#%#**#######*++*##*-::*#**##+:-::--=======+======+=====*+=+
@%@%@@@@@@%%%%@@@@@@%--+*+=:::=*+***=-+=%@@#++**#**++++*#@@%*+:*-****+:-::===+++++++=====+++++++#+=+
%%%%%%%%%%%%@@@@@@@@@=*++-=-::+++*#*+=*+@@%@%%%%%%%%%%%%%%@@++=*=****+::::-+++++===*====++-=+++*##+*
@@@@@@@@@@@@@@@@@@@@@=+*+=-::-++******#+%@@@%@@@@@@@@@@@@@@#=*+**%***+-=::=-====*+=+=++##*##%%###*##
@@@@@@@@@@@@@@@@@@@@@+**=*:::=+**%#%==#*%@@@%%@@%@%@@@@@@@%%=#++*%#***-=-:--==**###***#%#==%%@%####%
@@@@@@@@@@@@@@@@@@@@@***++::-+*##+-=%@@%#%@@#%@%%%@@@@@%%%%%=*@%==+#**+==-:*-#%@@@@@@@#=+%=*%%%%%@@@
@@@@@@@@@@@@@@@@@@@@@##=*-:=#++%%*+@@@%@@@@@###########**#%@@@@@%=%%#=+*==::++%@@@@@@%+*#%#*##%%#@@@
%%#=#%@@@@@@@@@@@@@@@#**%*=#@@@%%+@@%@%%@@@@@@%@@@@@@@%@@@@@@%%@@#*%@@@#==#-==%%%%%%%++#%#%##%###%%%
%%%%+*%%%@@@@@@@@@@%#*+*%@@@@%%%%%@%%%%%@@@@@@#%@@@@%##%@@@%%%%@@@#%%%@@@@%+=*#*##%%%+**#%%#%@%*##%%
%%#*+++=%%%%%%%@@%++%@@@@%%%%%%%%%%%%%%%%@@@@%*#%%@@@##%@@@@%%%%@@%%%%@%%%@@@%*=+##%%##*+#@%%%+*#%%%
%%%##%#+=######=*@@@@@@%%%%%%%%%%%%%%%%%%%@@@@*#%%%@@**#@@@%%%%%%@@%%%%%%%%%%%%%@#==%%%%***%#++%%##*
#%#**%%*++****%@@@@@%%%%%%%%%%%%%%%%%%%%%%%@@@*#***#%**#@@%%%%%%@@@@%%%%%%%%%%%%%%%%*#*%%%#%%#***%##
*%###%#***#*%@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%@@*%@@@@@##%@%%%%%%%@@@@%%%%%%%%%%%%%%%%@@%=%@%%@@##%###

"""