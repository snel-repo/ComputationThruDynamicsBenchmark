from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()
setup(
    name="ctd",
    version="1.0",
    install_requires=requirements,
    packages=find_packages(),
)
