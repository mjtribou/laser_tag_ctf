# setup.py
from setuptools import setup

setup(
    name="LaserTagCTF",
    version="0.1.1",
    options = {
        "build_apps": {
            "gui_apps":     {"Client": "client.py"},
            "console_apps": {"Server": "server.py"},
            "include_patterns": ["common/**","game/**","configs/**","models/**","README.md"],
            "exclude_patterns": ["**/__pycache__/**","**/*.pyc","runs/**"],
            "plugins": ["pandagl","p3openal_audio"],
            "platforms": ["manylinux2014_x86_64","win_amd64","macosx_11_0_arm64"],
            "log_filename": None,
        }
    }
)
