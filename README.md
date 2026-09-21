# Subject-verb agreement circuits in GPT-2

Term paper code. Finds the heads that do subject-verb agreement, then breaks them with
a distractor noun ("the key to the cabinets is"). Fails in Small and Medium both.

    pip install -r requirements.txt
    python run_experiment.py --model gpt2
    python run_experiment.py --model gpt2-medium --device cuda
    python compare_circuits.py results/gpt2 results/gpt2-medium
    python make_figures.py

Run in that order. Small works on CPU, Medium I ran on a 1660 Ti.

Edit src/dataset.py for the sentences, src/circuit_analysis.py for the method.
results/ is committed. figures/ isn't, make_figures.py rebuilds it.
