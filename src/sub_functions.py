import os

def function_get_project_root():
    """
    Get the root directory of the project.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
