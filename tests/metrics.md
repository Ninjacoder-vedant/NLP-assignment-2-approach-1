# Relative-Energy Anisotropy ($A_{\mathrm{RE}}$)

The core idea of this metric is extremely simple:

> "How spread out are my real embeddings compared with how spread out they would be if their directions were perfectly random?"

The metric is

$$
A_{\mathrm{RE}} = 1 - \frac{\mathbb{E}\left[\|X-X'\|\right]}{\mathbb{E}\left[\|X^\circ-X^{\circ\prime}\|\right]}
$$

## Intuition

Forget the notation for a moment. Think of your embeddings as points in a very high-dimensional space.

Suppose you have embeddings like this:

- one point corresponds to one sentence/token/representation
- every embedding has a length
- every embedding also points in some direction

For isotropy, we ideally want the directions to be spread uniformly around the entire space, rather than most embeddings pointing toward some preferred region.

The metric creates an imaginary "perfectly directionally random" version of your data.

For every real embedding, it does this conceptually:

$$
\text{real embedding} = \text{its length} + \text{its direction}
$$

It keeps the same length, but replaces its direction by a completely random direction:

$$
X^\circ = R\,U
$$

where

- $R = |X|$ is the original embedding length
- $U$ is a uniformly random direction

so $X^\circ$ has the same length statistics as your real embeddings, but no preferred direction.

### Why keeping the lengths matters

This is important.

We are not comparing your embeddings with some artificial unit sphere where every vector has length 1.

If your real vectors have lengths like

$$
3,\quad 5,\quad 5,\quad 8,\quad 12
$$

the reference also has lengths

$$
3,\quad 5,\quad 5,\quad 8,\quad 12
$$

Only their directions are randomized.

So differences in vector magnitude do not unfairly get called anisotropy.

## Numerator and denominator

Now look at the numerator:

$$
D = \mathbb{E}\|X-X'\|
$$

This means:

> Pick two real embeddings and measure the Euclidean distance between them. Do this for many pairs and take the average.

Suppose the average real distance is:

$$
D = 7
$$

Then calculate the same thing for the perfectly direction-randomized reference:

$$
B = \mathbb{E}\|X^\circ-X^{\circ\prime}\|
$$

Suppose:

$$
B = 10
$$

Then

$$
A_{\mathrm{RE}} = 1 - \frac{7}{10} = 0.3
$$

Interpretation:

The real embeddings are only $70\%$ as separated as they would be under rotational symmetry.

So there is a

$$
30\%
$$

relative loss of spread, which the metric calls anisotropy.

That does not mean "30% of dimensions are unused." The document explicitly warns against interpreting it that way.

## Mental model

The easiest mental model is this:

- Imagine throwing thousands of arrows from the origin.
- For a perfectly isotropic representation, arrows point equally in every possible direction.
- The real data may instead have many arrows pointing roughly toward the same region.
- When directions cluster, the endpoints of the arrows tend to be closer to each other.

Therefore

$$
\text{real average distance} < \text{random-direction average distance}.
$$

That distance deficit is what $A_{\mathrm{RE}}$ measures.

So:

$$
A_{\mathrm{RE}} = 1 - \frac{\text{actual spread}}{\text{spread expected under perfect directional randomness}}
$$

This one sentence is probably the best intuitive definition.

## Extreme cases

Some extreme examples make it clearer.

### Perfect isotropy

If your embeddings are already perfectly rotationally symmetric:

$$
D = B
$$

therefore

$$
A_{\mathrm{RE}} = 1 - \frac{B}{B} = 0.
$$

So

$$
\boxed{A_{\mathrm{RE}} = 0}
$$

means perfect isotropy at the population level. In fact, the proposed metric has the strong property that $A_{\mathrm{RE}} = 0$ iff the full distribution is rotationally symmetric about the origin.

### Total collapse

At the opposite extreme, suppose every embedding is exactly the same vector:

$$
X = X' = v.
$$

Then every pair has distance

$$
\|X-X'\| = 0.
$$

So

$$
D = 0
$$

and therefore

$$
A_{\mathrm{RE}} = 1 - \frac{0}{B} = 1.
$$

Thus

$$
\boxed{A_{\mathrm{RE}} = 1}
$$

means maximum possible collapse: every embedding has become the same point.

## Isotropy score

You can also express the complementary score:

$$
I_{\mathrm{RE}} = 1 - A_{\mathrm{RE}} = \frac{D}{B}.
$$

This is the isotropy score.

So roughly:

| Situation                      | $A_{\mathrm{RE}}$ (anisotropy) | $I_{\mathrm{RE}}$ (isotropy) |
| ------------------------------ | ------------------------------ | ---------------------------- |
| Perfectly isotropic            | 0                              | 1                            |
| Some directional concentration | e.g. 0.25                      | 0.75                         |
| Strong concentration           | e.g. 0.8                       | 0.2                          |
| Every vector identical         | 1                              | 0                            |

### A warning on interpretation

But there is one very important warning.

Do not read

$$
I_{\mathrm{RE}} = 0.8
$$

as

> "the embeddings are 80% isotropic."

It only means

$$
\frac{\text{their average pairwise separation}}{\text{separation expected from randomized directions}} = 0.8.
$$

That is a precise mathematical ratio, not a percentage of dimensions, directions, or information.

## Implementation

The actual implementation uses

$$
\widehat{A}_{\mathrm{RE}} = 1 - \frac{\sum_{i<j}\|x_i-x_j\|}{\sum_{i<j} b_d(r_i, r_j)}.
$$

Here the upper part is straightforward:

$$
\sum_{i<j}\|x_i-x_j\|
$$

means calculate the distance for every pair of your real embeddings.

The weird-looking function

$$
b_d(r_i, r_j)
$$

simply answers:

> "If I had two vectors of lengths $r_i$ and $r_j$, but pointed them in completely random directions in $d$ dimensions, what average distance should I expect between them?"

So you do not actually need to randomly generate thousands of fake embeddings. The expected random distance can be calculated mathematically.

The whole pipeline is therefore basically:

$$
\boxed{\text{Real pairwise spread} \quad \text{vs} \quad \text{Expected pairwise spread under random directions}}
$$

Then:

$$
\boxed{A_{\mathrm{RE}} = \frac{\text{spread that has been lost}}{\text{spread expected under isotropy}}}
$$

## Why this beats mean cosine similarity

There is also a particularly nice reason this metric is stronger than simply averaging cosine similarity.

Cosine might say:

> "On average, are vectors pointing similarly?"

This metric asks:

> "Does the entire geometric distribution behave like something rotationally symmetric?"

Because it uses ordinary Euclidean distance,

$$
\|u-v\| = \sqrt{2 - 2u^\top v},
$$

the square root makes it a nonlinear function of angular similarity. That allows the energy-distance theory behind it to detect distributional differences that a simple mean cosine can lose.

## Takeaway

For your LLM anisotropy research, the intuition I would keep in your head is:

> Take the geometry the model actually produced. Keep every embedding's length exactly as it is. Randomize only where the vectors point. Then ask how much more spread-out that ideal randomized cloud would be than the real cloud. That missing spread is the anisotropy score.

## Limitation

One major limitation remains: in extremely high dimensions, some very structured anisotropic distributions can still produce surprisingly tiny scores. The document's coordinate-axis example gives approximately $A_{\mathrm{RE}} = 0.000041$ at $d = 4096$, despite the distribution not being rotationally symmetric. So this is mathematically exact as a characterization, but a small numerical score does not automatically mean practically negligible anisotropy in high-dimensional LLM spaces.
