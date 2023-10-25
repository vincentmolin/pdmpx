# pdmpx
PDMPs in JAX!

There are some examples in the [examples folder](examples/).




Checklist:
 * [x] Refactor over thinning timer (50%)
 * [x] Add examples from poster
   * Simple Cold BPS impl.
 * [x] Add tests
----
 * [x] Quadratic approx timer
   * [x] Test cpu impl. vs numpy
   * [x] Test jax impl. vs cpu / numpy
   * [ ] Test n-th dir deriv
 * [ ] Add OSCN bounce kernel
 * [ ] gh-pages / docs?
 * [x] (Covered in example!) Might be more JAXy to only ever expose the ```get_next_event``` method for a PDMP.
       Probably simplifies the context handling everywhere.

