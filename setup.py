from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, 'r') as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name='linesight',
    version='3.0.1',
    description='Trackmania AI with reinforcement learning.',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10.1,<3.12.0',
    install_requires=read_requirements('requirements_pip.txt') + read_requirements('requirements_conda.txt'),
    packages=find_packages(include=["trackmania_rl", "config_files"]),
)
