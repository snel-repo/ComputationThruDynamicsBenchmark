from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()
requirements.append("DSA @ git+https://github.com/mitchellostrow/DSA.git@main#egg=DSA")
setup(
    name="ctd",
    version="1.0",
    install_requires=requirements,
    packages=find_packages(),
)
