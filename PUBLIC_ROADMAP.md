# RuleChef Roadmap

RuleChef is early and evidence-driven. This roadmap separates near-term
correctness work from research directions whose APIs and value are not yet
established. It intentionally has no delivery dates.

## Vision

RuleChef learns inspectable local behavior from examples and model outputs.
The near-term goal is trustworthy selective execution: serve a local result
only when its evidence and support justify it, validate the output, and defer
to the configured model or human fallback otherwise.

We are also exploring whether symbolic, linguistic, and semantic methods can
handle complementary regions of a workload. Those directions will be shaped by
benchmarks and contributor experience rather than committed in advance.

## Now: trustworthy learning and routing

The
[v0.3 — Trustworthy Learning and Routing milestone](https://github.com/KRLabsOrg/rulechef/milestone/1)
contains correctness work that is useful regardless of later experiments:

- [split before synthesis and preserve evaluation roles](https://github.com/KRLabsOrg/rulechef/issues/32);
- [support- and confidence-thresholded fallback](https://github.com/KRLabsOrg/rulechef/issues/8);
- [fail closed on schema-invalid local output](https://github.com/KRLabsOrg/rulechef/issues/33);
- [evaluate scalar transformation fields](https://github.com/KRLabsOrg/rulechef/issues/34);
- [separate routed, agreeing, and correctly routed coverage](https://github.com/KRLabsOrg/rulechef/issues/35);
- [align package, API, and module versions](https://github.com/KRLabsOrg/rulechef/issues/36).

The milestone is intentionally about measurement and routing foundations, not a
new rule language.

## Exploring: richer local rules

The undated
[Research — Richer Local Rules milestone](https://github.com/KRLabsOrg/rulechef/milestone/2)
is a collaborator playground. Its issues are experiments and design discussions,
not committed release features or stable APIs.

### RuleIR direction

RuleIR is a working name for a possible minimal common representation:

~~~text
predicate -> guard -> action -> evidence
policy    -> ordered rules + conflict behavior + fallback
~~~

Predicates could include regex, structured-field checks, spaCy token/dependency
patterns, and semantic prototypes. A useful RuleIR should make these artifacts
composable and inspectable without forcing an immediate rewrite of current
Rule objects.

The open
[RuleIR RFC issue](https://github.com/KRLabsOrg/rulechef/issues/37)
is deliberately design-only. A recursive general-purpose DSL, signed deployment
bundles, and a large schema migration are out of scope until smaller experiments
justify them.

### Semantic/vector prototypes

[The semantic-prototype research spike](https://github.com/KRLabsOrg/rulechef/issues/38)
asks whether positive exemplars, hard negatives, similarity, and class margins
can selectively handle regions that regex misses. It starts benchmark-local and
compares against regex, TF-IDF, embedding-classifier, and fallback baselines
before proposing a public API.

### spaCy

Existing spaCy work is grouped into the same research milestone:

- [evaluate dependency rules against regex](https://github.com/KRLabsOrg/rulechef/issues/23);
- [improve and measure spaCy pattern synthesis validity](https://github.com/KRLabsOrg/rulechef/issues/24);
- [add a dependency-rule example and documentation](https://github.com/KRLabsOrg/rulechef/issues/25).

Contributions that produce reproducible comparisons, failure analyses, or small
design notes are valuable even when the result is negative.

## Later, if validated

Depending on benchmarks and user feedback, later work may include:

- minimal composition of heterogeneous predicates;
- ruleability analysis and artifact-family comparison;
- stronger decision/evidence traces;
- shadow evaluation and monitoring;
- compact local experts where rules are not the best fit.

These are possible directions, not promised features.

## How this roadmap changes

- Near-term correctness issues can move directly into a release milestone.
- Research issues should produce evidence or a design decision before becoming
  public API work.
- Simpler baselines and negative results can narrow or stop a direction.
- New product and deployment commitments should follow actual user or
  design-partner evidence.
