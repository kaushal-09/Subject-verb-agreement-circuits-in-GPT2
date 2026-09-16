# Subject-verb agreement circuits in GPT-2

Code for a replication and extension of Africa (2025), *Identifying a Circuit for
Verb Conjugation in GPT-2* ([arXiv:2506.22105](https://arxiv.org/abs/2506.22105)).

We find the attention heads most responsible for subject-verb agreement, then test
whether that circuit still works on **agreement attraction** — sentences where a noun
of the wrong number sits between the subject and the verb, as in *the key to the
cabinets is*. It does not, in either GPT-2 Small or GPT-2 Medium.

## Setup

    pip install -r requirements.txt

## Running

    python run_experiment.py --model gpt2
    python run_experiment.py --model gpt2-medium --device cuda
    python compare_circuits.py results/gpt2 results/gpt2-medium
    python make_figures.py

The first two write per-model results to `results/<model>/`, the third the cross-model
comparison, the fourth figures and LaTeX tables to `figures/`. GPT-2 Small runs on CPU
in a few minutes; the committed results came from a GTX 1660 Ti.

## Layout

    src/dataset.py           130 minimal pairs across six conditions
    src/circuit_analysis.py  activation patching, circuit-restricted evaluation
    run_experiment.py        find a circuit in one model and score it
    compare_circuits.py      compare two models' circuits
    make_figures.py          figures and tables

## Method notes

**Patching is averaged over 30 prompt pairs.** A single clean/corrupted pair gives a
maximum recovery of about 1% for any head, which is noise — not enough to rank heads by.

**Circuit size is a fraction of total heads, not a fixed count.** Africa reports 12 of
144 heads for GPT-2 Small, so we keep 8.33% for every model: 12 heads for Small, 32 for
Medium. Fixing the count at 12 would hand the larger model a circuit less than a third
as dense and confound scale with circuit size.

**Two ablation methods, mean reported as primary.** Evaluating a circuit means ablating
the other ~90% of heads. Zeroing that many pushes the residual stream far outside the
distribution the model was trained on; mean-ablation keeps each head's average
contribution while removing its input-dependence. Zero-ablation results proved sensitive
to stimulus-set size where mean-ablation results did not. Africa uses resample ablation,
which we do not implement — mean is the nearer of our two to it.

**Conditions are not equally hard.** Simple, pronoun and negation items are all solvable
by agreeing with the nearest preceding noun. Attraction items are the only ones where
that heuristic and correct subject tracking disagree, which is why they are the
interesting case.
