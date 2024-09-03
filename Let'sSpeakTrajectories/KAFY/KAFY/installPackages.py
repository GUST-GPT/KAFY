"""
Small module responsible to install packages that are imported
"""

import subprocess
import sys


# Function to install a package if it's not already installed.
def install_package(package):
    """
    responsible to install a package that is imported
    to make sure the pipeline is runnable anywhere
    """
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
