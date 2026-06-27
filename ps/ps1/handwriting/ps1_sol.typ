#set page(numbering: "1")
#set heading(numbering: "1.a")
#set text(font: "New Computer Modern", size: 12pt)
#align(left, text(17pt)[
  *Problem Set #1: Supervised Learning*
])

= Linear Classifiers (logistic regression and GDA)

(a) Given the average empirical loss for logistic regression
$
  J(theta) = -1/m sum_(i=1)^m y^((i)) log(h_theta (x^((i)))) + (1 - y^((i)))log(1 - h_theta (x^((i))))
$
where $y^((i)) in {0, 1}, h_theta (x) = g(theta^T x)$ and $g(z) = frac(1, 1 + e^(-z))$

Then, the gradient of $J(theta)$ is
$
  gradient_theta J(theta)
  &= 1/m sum_(i=1)^m [frac(y^((i)), h_theta (x^((i)))) - frac((1 - y^((i))), 1 - h_theta (x^((i))))] h_theta (x^((i))) (1 - h_theta (x^((i))) x^((i)) \
  &= 1/m sum_(i=1)^m [y^((i)) - h_theta (x^((i)))] x^((i)) \
  &= 1/m mat(x^((1)), x^((2)), dots.h, x^((m))) vec(y^((1)) - h_theta (x^((1))), dots.v, y^((m)) - h_theta (x^((m)))) \
  &= 1/m X^T (y - h_theta (X))
$
where $X = vec(x^((1)), dots.v, x^((m)))$ and $h_theta (X) = frac(1, 1 + e^(-X theta))$

And the Hessian of $J(theta)$ is
$
  H & = 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] x^((i)) (x^((i)))^T \
    & = 1/m X^T "diag"{h_theta (x^((1))) [1 - h_theta (x^((1)))], dots.h, h_theta (x^((m))) [1 - h_theta (x^((m)))]} X \
    & = X^T D X
$
where $D = "diag"{h_theta (x^((i))) [1 - h_theta (x^((i)))]}_(i=1)^m$

$forall z in RR^n, z^T H z
&= sum_(k=1)^n sum_(l=1)^n z_k H_(k l) z_l \
&= sum_(k=1)^n sum_(l=1)^n z_k [1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] x_k^((i)) x_l^((i))] z_l \
&= 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] sum_(k=1)^n sum_(l=1)^n z_k x_k^((i)) x_l^((i)) z_l \
&= 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] (z^T x^((i)))^2 >= 0$

Thus $J(theta)$ is convex and hence if there is any critical point, it is its global minimum.

(c)
$
  &p(y=1 | x; Phi, mu_0, mu_1, Sigma) \
  &= frac(p(x|y=1)p(y=1), p(x)) \
  &= frac(p(x|y=1)p(y=1), p(x|y=1) p(y=1) + p(x|y=0) p(y=0)) \
  &= frac(1, 1 + frac(p(x|y=0) p(y=0), p(x|y=1) p(y=1))) \
  &= frac(1, 1 + frac(exp(-1/2 (x - mu_0)^T Sigma^(-1) (x - mu_0)) (1-Phi), exp(-1/2 (x - mu_1)^T Sigma^(-1) (x - mu_1)) Phi)) \
  &= frac(1, 1 + exp(-1/2 (x - mu_0)^T Sigma^(-1) (x - mu_0) + 1/2 (x - mu_1)^T Sigma^(-1) (x - mu_1) + log((1-Phi)/Phi))) \
  &= frac(1, 1 + exp(-((mu_0 - mu_1)^T Sigma^(-1) x - 1/2 mu_1^T Sigma^(-1) mu_1 + 1/2 mu_0^T Sigma^(-1) mu_0 - log((1-Phi)/Phi)))) \
  &= frac(1, 1 + exp(-(theta^T x + theta_0)))
$
where $theta = Sigma^(-1) (mu_1 - mu_0)$ and $theta_0 = -1/2 mu_1^T Sigma^(-1) mu_1 + 1/2 mu_0^T Sigma^(-1) mu_0 - log((1-Phi)/Phi)$

(d)
$
  & l(Phi, mu_0, mu_1, sigma) \
  & = log product_(i=1)^m p(x^((i))|y^((i));mu_0, mu_1, sigma) p(y^((i)); Phi) \
  & = sum_(i=1)^m 1{y^((i))=1} log(x^((i))|y^((i))=1) + 1{y^((i))=1} log(p(y^((i))=1)) \
  & + 1{y^((i))=0} log(x^((i))|y^((i))=0) + 1{y^((i))=0} log(p(y^((i))=0)) \
  & = sum_(i=1)^m 1{y^((i))=1} (- frac((x^((i))-mu_1)^2, 2 sigma^2) - log(sqrt(2 pi) sigma)) + 1{y^((i))=1} log(Phi) \
  & + 1{y^((i))=0} (- frac((x^((i))-mu_0)^2, 2 sigma^2) - log(sqrt(2 pi) sigma)) + 1{y^((i))=0} log(1-Phi) \
$

$"FOCs:" \
& frac(partial l, partial Phi) = sum_(i=1)^m 1{y^((i))=1} 1/Phi - 1{y^((i))=0} 1/(1-Phi) = 0 \
& => 1/Phi sum_(i=1)^m 1{y^((i))=1} = 1/(1-Phi) sum_(i=1)^m 1{y^((i))=0} \
& => 1/Phi -1 = frac(sum_(i=1)^m 1{y^((i))=0}, sum_(i=1)^m 1{y^((i))=1}) \
& => 1/Phi = frac(m, sum_(i=1)^m 1{y^((i))=1}) \
& => Phi = 1/m sum_(i=1)^m 1{y^((i))=1}$

$
  frac(partial l, partial mu_0) = sum_(i=1)^m 1{y^((i))=0} (x^((i))-mu_0) / sigma^2 = 0 => mu_0 =
  frac(sum_(i=1)^m 1{y^((i))=0} x^((i)), sum_(i=1)^m 1{y^((i))=0})
$

$"Similarly," mu_1 = frac(sum_(i=1)^m 1{y^((i))=1} x^((i)), sum_(i=1)^m 1{y^((i))=1})$

$
  &frac(partial l, partial sigma) = sum_(i=1)^m 1{y^((i))=1} (frac((x^((i)) - mu_1)^2, sigma^3) - 1/sigma) + 1{y^((i))=0} (frac((x^((i)) - mu_0)^2, sigma^3) - 1/sigma)= 0 \
  &=> 1/sigma^3 (sum_(i=1)^m 1{y^((i))=1} (x^((i)) - mu_1)^2 + 1{y^((i))=0} (x^((i)) - mu_0)^2) - m/ sigma = 0 \
  &=> sigma^2 = 1/m (sum_(i=1)^m 1{y^((i))=1} (x^((i)) - mu_1)^2 + 1{y^((i))=0} (x^((i)) - mu_0)^2)
$

(f)
#figure(
  image("../reports/figures/ds1_train.jpg", height: 8cm),
  caption: [Training dataset 1],
)

(g)
#figure(
  image("../reports/figures/ds2_train.jpg", height: 8cm),
  caption: [Training dataset 2],
)

GDA seem to perform worse on dataset 1, which is right-skewed

(h)
$T(x)="sign"(x) log(|x|+1)$
#figure(
  image("../reports/figures/ds1_train_trans.jpg", height: 8cm),
  caption: [Training dataset 1 with transformed inputs],
)

= Incomplete, Positive-Only Labels

(a)
$
  p(t^((i))=1 | y^((i))=1) =1 => p(t^((i))=1 | y^((i))=1, x^((i))) = 1
$
$
  &p(y^((i))=1 | x^((i))) \
  & quad = p(y^((i))=1 | t^((i))=1, x^((i))) p(t^((i))=1 | x^((i))) + p(y^((i))=1 | t^((i))=0, x^((i))) p(t^((i))=0 | x^((i))) \
  & quad = p(y^((i))=1 | t^((i))=1) p(t^((i))=1 | x^((i))) + p(t^((i))=0 | y^((i))=1, x^((i))) p(y^((i))=1 | x^((i))) \
  \
  & => p(t^((i))=1 | y^((i))=1, x^((i))) p(y^((i))=1 | x^((i))) = p(y^((i))=1 | t^((i))=1) p(t^((i))=1 | x^((i))) \
  & => p(y^((i))=1 | x^((i))) = frac(p(y^((i))=1), p(t^((i))=1)) p(t^((i))=1 | x^((i))) \
  & => p(t^((i))=1 | x^((i))) = p(y^((i))=1 | x^((i))) / alpha
$
where $alpha = frac(p(y^((i))=1), p(t^((i))=1))$

(b)
$
  forall x^((i)) in V_+, h(x^((i))) approx p(y^((i))=1 | x^((i))) = alpha . p(t^((i))=1 | x^((i))) approx alpha.1 = alpha
$

(c)






