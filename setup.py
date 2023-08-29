from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='flows_new',
    version='0.1',
    description='A shot at re-writing the flows repository.',
    install_requires=requirements,
    packages=find_packages(include=['flows_new'])
)