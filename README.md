# Subject-verb agreement circuits in GPT-2

Finds the attention heads that handle subject-verb agreement in
GPT-2, then checks whether those heads still work once you put a distractor noun
between the subject and the verb ("the key to the cabinets is"). They don't. The
circuit keeps negation, pronouns and plurals at full accuracy and then gets every
single distractor sentence wrong, in Small and Medium both, so it is probably just
agreeing with the nearest noun rather than tracking the actual subject.

    pip install -r requirements.txt
    python run_experiment.py --model gpt2
    python run_experiment.py --model gpt2-medium --device cuda
    python compare_circuits.py results/gpt2 results/gpt2-medium
    python make_figures.py

When we them in that order, each one reads what the last one wrote. Small is fine to run on the CPU
and takes a few minutes, Medium one I ran on a 1660 Ti graphics card.

run_experiment.py does the whole pipeline for one model: builds the 130 test
sentences, runs the patching sweep to score every head, keeps the top 8.33% as the
circuit, then re-runs the model with everything else ablated to see what survives.
Output goes to results/<model>/. compare_circuits.py takes two of those folders and
compares them, which is where the depth profile numbers come from.
make_figures.py draws the plots and writes tables.tex.

The two files src/dataset.py for the sentences and
src/circuit_analysis.py for the method. 
results/ is committed so you can read
the numbers without rerunning anything. 
figures/ is gitignored because make_figures.py
rebuilds it from results/ anyway.
