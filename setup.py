from setuptools import setup

install_requires = [
    # "gurobipy",  # install this manually
    # "alib",
    # "basemap",  # install this manually
    "click",
    "matplotlib",
    "numpy",
]

setup(
    name="vnep-approx",
    # version="0.1",
    packages=["vnep_approx"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "vnep-approx = vnep_approx.cli:cli",
        ]
    }
)
