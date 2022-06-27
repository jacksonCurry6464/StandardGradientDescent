from setuptools import setup, find_packages

setup(
    name='gdtest',
    version='1.0',
    description='This package implements gradient descent and tests it on an example problem',
    author='Jackson Curry',
    author_email='jacksoncurry6464@gmail.com',
    install_requires=['numpy', 'matplotlib'],
    packages=find_packages(),

)