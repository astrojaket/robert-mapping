# Student glossary

Use this page when a word in a guide, YAML file, plot, or result is not clear.
You do not need to learn all of these words before the first run.

## Analysis words

**Baseline**
: A slow change in measured flux that is not caused by the planet map.

**BIC — Bayesian Information Criterion**
: A score that balances fit quality against model complexity. Lower is better.
  In this project, a negative `Delta BIC = BIC(map) - BIC(uniform)` favours the
  map.

**Cadence**
: The time between two measurements.

**Chain**
: One independent sampler run. Several chains help us check that the sampler
  found the same result more than once.

**Corrected or detrended light curve**
: A light curve after an instrument trend has been removed.

**Divergence**
: A warning that the sampler could not follow part of the probability shape
  correctly. A production result should normally have zero divergences.

**ESS — effective sample size**
: An estimate of how many independent samples the correlated saved samples
  contain. Larger is better.

**Injection and recovery**
: A test in which we add a known signal to synthetic or real residual data and
  check if the code finds it.

**Likelihood**
: The rule that scores how probable the data are for one set of model values.

**NUTS — No-U-Turn Sampler**
: The algorithm that makes posterior samples. It changes its step length and
  path length during warm-up.

**Posterior**
: The probability distribution after the model combines the data with the
  prior.

**Prior**
: The allowed range and probability for a model value before this data set is
  fitted.

**Regressor**
: A measured quantity, such as detector position, that can explain an
  instrument trend.

**Residual**
: Data minus model. A residual plot shows what the model did not explain.

**R-hat**
: A check that compares sampler chains. A value close to 1 is good. The study
  guides give the required threshold for each run.

**Systematics**
: Changes in measured flux that come from the instrument, telescope, or data
  reduction instead of the planet.

**Warm-up**
: Sampler steps used to tune NUTS. These steps are not saved as posterior
  draws.

## Light-curve and map words

**Eclipse map**
: A two-dimensional estimate of planet brightness made from the way different
  planet regions disappear behind the star and return.

**Full phase curve**
: Brightness measured through most or all of one planet orbit, including the
  changing view of the planet.

**Hot spot**
: The brightest fitted region on the planet map.

**Hot-spot offset**
: The longitude angle between the hot spot and the sub-stellar point. The
  sub-stellar point faces the star.

**Latitude and longitude**
: Map coordinates. Eclipse data usually constrain longitude better than
  latitude.

**ppm — parts per million**
: A small relative-flux unit. `100 ppm` is `0.0001` in normalized flux.

**Uniform map**
: A comparison planet with the same brightness at every location.

**White noise**
: Random measurement error with no memory. The error at one time does not
  make a similar error at the next time more likely.

**Time-correlated noise**
: Residual error with memory. Nearby measurements can have similar errors.
  The YAML value `noise_model: ou` selects one mathematical form called an
  Ornstein–Uhlenbeck model. Its amplitude gives the strength of this noise.
  Its timescale gives how quickly the memory fades.

## Software words

**JAX**
: The numerical library that evaluates and differentiates the model.

**NumPyro**
: The inference library that runs NUTS with JAX.

**YAML**
: The readable text format that stores the choices for one analysis. YAML is
  configuration, not executable Python code.
