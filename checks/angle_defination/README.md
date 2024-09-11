# Checks of different way of angle calculation

`test.py` provide a preprocessor implement use
[decayangle](https://github.com/kaihabermann/decayangle/) package to calculate
angle.

It show consistant results on amplitude and alignment angle. But, due to the
different conversion, some transforms are required to get the same results. For
example, the helicity angle have the transform $\phi\rightarrow -\phi$ and
$\theta \rightarrow -\theta$.

In addtion, if some one change the particle order, it would require other
transform.
