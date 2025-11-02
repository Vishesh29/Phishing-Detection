'''
The setup.py file is an essential part of packaging and distributing Python projects. 
It is used to setuptools ( or disutils in older python versions) to define the configuration
of your project, such as its metadata, dependencies, and entry points.
'''

from setuptools import find_packages, setup 
# consider init file inside directory as packages
from typing import List

def get_requirements()->List[str]:
    '''
    This functions will return the list of requirements
    '''
    requirement_list:List[str] = []
    try:
        with open('requirements.txt') as file:
            # Read lines from file
            lines = file.readlines()
            ## Process each line
            for line in lines:
                requirement = line.strip()
                # Ignore comments and empty lines ie -e .
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
        
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists in the project directory.")
    
    return requirement_list


setup(
    name='mlops_project',
    version='0.0.1',
    author='Vishesh',
    author_email='visheshsaxena29@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
    )