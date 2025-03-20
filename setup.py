from setuptools import setup, find_packages

setup(
    name="torch-paqm",
    version="0.0.1",
    author="Bernardo Vieira de Miranda",
    install_requires=[
        "torch",
        "torchaudio",
        "scipy",
    ],
    extras_require={
        "tests": ["pytest", "matplotlib"],
    },
    packages=find_packages(include=["./src/paqm*"]),
)
