from pathlib import Path

def get_project_root():
    """Returns the path to the project root directory."""
    # The script is in src/, so parent.parent is the project root
    return Path(__file__).parent.parent
