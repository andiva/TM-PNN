from setuptools import setup, find_packages
import tm_pnn as tm

setup(
    name='tm_pnn',
    python_requires='>3.6',
    version=tm.__version__,
    packages=find_packages(),
    author='Andrei Ivanov',
    author_email='05x.andrey@gmail.com',
    url='https://github.com/andiva/TM-PNN.git',
    test_suite='tests',
)