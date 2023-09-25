from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    install_requires = file.read().splitlines()

setup(
    name="fixed_point_finder",
    author="Systems Neural Engineering Lab",
    version="0.0.0dev",
    install_requires=install_requires,
    packages=find_packages(),
)
