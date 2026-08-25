"""Run the full implemented verification suite and print a summary. See README.md
for what is and is not covered."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (
    exp1_baseline, exp2_payload_sweep, exp3_interaction_force,
    exp4_contact_stiffness_step, exp5_flagship_reroute, exp6_severity_sweep,
    ablation_batch, timing_benchmark,
)

SECTIONS = [
    ("Sanity checks (dynamics + planner)", None),
    ("Experiment 1: baseline trajectory, no regression check", exp1_baseline.run),
    ("Experiment 2: payload sweep, conservatism", exp2_payload_sweep.run),
    ("Experiment 3: interaction force, detection lead time", exp3_interaction_force.run),
    ("Experiment 4: contact-stiffness transition, known in advance", exp4_contact_stiffness_step.run),
    ("Experiment 5: flagship reroute", exp5_flagship_reroute.run),
    ("Experiment 6: severity sweep across Level 0-4", exp6_severity_sweep.run),
    ("Ablation batch A1-A5", ablation_batch.run),
    ("Real-time timing benchmark (Sec. VIII-J)", timing_benchmark.run),
]

if __name__ == "__main__":
    import subprocess
    print("=" * 70)
    print("Dynamics + planner correctness tests")
    print("=" * 70)
    subprocess.run([sys.executable, os.path.join("tests", "test_dynamics.py")], check=True)
    subprocess.run([sys.executable, os.path.join("tests", "test_planner.py")], check=True)

    for title, fn in SECTIONS:
        if fn is None:
            continue
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        fn()
