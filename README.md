# pdmpx
PDMPs in JAX

Checklist (Tuesday lunch):
 * [ ] Refactor over thinning timer
 * [ ] Add examples from poster, BPS
   * Simple Cold BPS impl.
 * [x] Add tests
 * [ ] Finish typings, add comments for the most exposed

Checklist (Additional):
 * [ ] Quadratic approx timer
   * Test cpu impl. vs numpy
   * Test jax impl. vs cpu / numpy
   * Test n-th dir deriv
 * [ ] Add OSCN bounce kernel
 * [ ] gh-pages / docs?
 * [ ] Might be more JAXy to only ever expose the ```get_next_event``` method for a PDMP.
       Probably simplifies the context handling everywhere.

