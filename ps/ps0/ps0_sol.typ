#set page(numbering: "1")
#set heading(numbering: "1.a")

#set text(font: "New Computer Modern", size: 12pt)
#set par(justify: true)

#align(left, text(17pt)[
  *Problem Set #0: Linear Algebra and Multivariable Calculus*
])

= Gradients and Hessians

(a) $f(x) = 1/2 x^T A x + b^T x$. What is gradient $f(x)$?

$x^T A x = sum_(i, j)a_(i j) x_i x_j => partial / (partial x_i) x^T A x = 2 sum_j a_(i j) x_j => gradient(x^T A x) = vec(
  2
  sum_j a_(1 j) x_j, dots.v, 2 sum_j a_(n j) x_j
) = 2 A x \
b^T x = sum_i b_i x_i => partial / (partial x_i) b^T x = b_i => gradient(b^T x) = vec(b_1, dots.v, b_n) = b \
"Thus," gradient f(x) = A x + b$

(b) $f(x) = g(h(x)) "where" g: RR -> RR "and" h: RR^n -> RR$.What is $gradient f(x)$?

$frac(partial g(h(x)), partial x_i) = frac(partial g(h(x)), partial h(x)) frac(partial h(x), partial x_i) = g'(h(x)) frac(partial h(x), partial x_i) => gradient f(x) = g'(h(x)) gradient h(x)$

(c) $f(x) = 1/2 x^T A x + b^T x$. What is $gradient^2 f(x)$?

$frac(partial, partial x_i partial x_j) x^T A x = 2 a_(i j) => gradient^2 (x^T A x) = 2A \
frac(partial, partial x_i partial x_j) b^T x = 0 => gradient^2 (b^T x) = 0 \
=> gradient^2 f(x) = A$

(d) $f(x) = g(a^T x)$. What are $gradient f(x) "and" gradient^2 f(x)?$

$gradient f(x) = g'(a^T x)a \
a^T x = sum_i a_i x_i => frac(partial f(x), partial x_i) = g'(a^T x) a_i => frac(partial f(x), partial x_i partial x_j) = g''(a^T x) a_i a_j => gradient^2 f(x) = g''(a^T x) a a^T$

= Positive definie matrices

(a) $z in RR^n. "Show" A = z z^T "is PSD."$

$forall x in RR^n, x^T A x = x^T z z^T x = (z^T x)^T (z^T x) = ||z^T x||_2^2 >= 0$

(b) $z in RR^n "is non-zero and" A = z z^T. "What are" ker(A) "and rank"(A)?$

$0 = A x = z z^T x = mat(
  z_1 z_1, z_1 z_2, dots.h, z_1 z_n;
  dots.v, dots.v, dots.down, dots.v;
  z_n z_1, z_n z_2, dots.h, z_n z_n
) vec(x_1, x_2, dots.v, x_n) = vec(z_1 sum_i z_i x_i, dots.v, z_n sum_i z_i x_i) \
z != 0 => z_i != 0 "for some" i => sum_i z_i x_i = 0
"Thus, " ker(A) = {x in RR^n: x^T z = 0} \
"rank"(A) = 1$

(c) $A in RR^(n times n) "is PSD and" B in RR^(m times n)$. Is $B A B^T$ PSD?

$forall x in RR^m, x^T B A B^T x = (B^T x)^T A (B^T x) >= 0 "since A is PSD" \
"Thus, " x^T B A B^T "is PSD"$

= Eigenvectors, eigenvalues, and the spectral theorem

(a) Show $A t^((i)) = lambda_i t^((i))$

$A = T Lambda T^(-1) => A T = T Lambda => A (t^((1)), dots.h, t^((n))) = (t^((1)), dots.h, t^((n))) mat(lambda_1, 0, dots.h, 0; 0, lambda_2, dots.h, 0; dots.v, dots.v, dots.down, dots.v; 0, 0, dots.h, lambda_i) \
=> (A t^((1)), dots.h, A t^((n))) = (lambda_1 t^((1)), dots.h, lambda_n t^((n))) => A t^((i)) = lambda_i t^((i))$


(b) Show that $A u^((i)) = lambda_i u^((i))$

$U^T U = I => U^T = U^(-1)$ \
In (a), take $T = U => A u^((i)) = lambda_i u^((i))$

(c) Show that if A is PSD, $lambda_i(A) >= 0, forall i$

$I = U^T U = vec(u^(1)^T, dots.v, u^(n)^T) (u^((1)) dots.h u^((n))) => u^(i)^T u^((i)) = 1, forall i \
A u^((i)) = lambda_i u^((i)) => u^(i)^T A u^((i)) = u^(i)^T lambda_i u^((i)) = lambda_i (A) \
"A is PSD" => lambda_i (A) >= 0$
