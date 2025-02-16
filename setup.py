from setuptools import find_packages, setup
from typing import List

HYPE_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirement libs mentioned in requirements.txt
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements] # replacing \n with empty and creating it as list

        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Ravi',
    author_email='rravii.r@gmail.com',
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'seaborn']  # we cannot provide all the libs like this, so we can read from requiremnts.txt
    install_requires=get_requirements('requirements.txt')
)
