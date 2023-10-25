# pdmpx
PDMPs in JAX!

Take a look at the examples in the [examples folder](examples/).




Checklist:
 * [x] ~~Refactor over thinning timer~~ 
   * [ ] add better tests for linear thinning timer
----
 * [x] Quadratic approx timer :rocket:
   * [x] Test cpu impl. vs numpy
   * [x] Test jax impl. vs cpu / numpy
   * [ ] Test n-th dir deriv
 * [ ] Add OSCN bounce kernel
 * [ ] ~~gh-pages / docs?~~
 * [x] (Covered in example!) Might be more JAXy to only ever expose the ```get_next_event``` method for a PDMP.
       Probably simplifies the context handling everywhere.

