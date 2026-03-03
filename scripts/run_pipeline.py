import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = [
    "scripts/01_parse.py",
    "scripts/02_extract.py",
    "scripts/03_build_graph.py",
    "scripts/04_embed.py",
]


def main():
    python = sys.executable
    for script in SCRIPTS:
        script_path = PROJECT_ROOT / script
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [python, str(script_path)],
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            print(f"\nERROR: {script} failed with code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}")
    print("\nTo launch the Q&A interface:")
    print(f"  {python} -m src.app")


if __name__ == "__main__":
    main()
