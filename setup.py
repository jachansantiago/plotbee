import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="plotbee",
    version="0.0.1",
    packages=setuptools.find_packages(),
    scripts=['scripts/pb'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    extra_requires={
        "tags": [f"apriltag @ file://localhost/{os.getcwd()}/apriltag#egg=apriltag"]
        },
    python_requires='>=3.6',
)
