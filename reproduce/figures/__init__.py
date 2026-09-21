"""The article's figures, from the released tables.

Each module draws one figure and can be run on its own. Nothing plotted is typed in:
every value is read from a table in ``../data`` when the figure is drawn.
"""
import importlib
import os

FIGURES = ("fig_F3", "fig_F4", "fig_F5", "fig_F6", "fig_F7", "fig_F8")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures")


def main():
    """Draw every figure, in a fresh process each, since the modules run at import."""
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    for name in FIGURES:
        subprocess.run([sys.executable, os.path.join(here, f"{name}.py")], check=True,
                       stdout=subprocess.DEVNULL)
    print(f"figures written: {sorted(f for f in os.listdir(OUT) if f.endswith('.pdf'))}")
