from setuptools import setup

install_requires = [
    'numpy'
]

packages = [
    'loadflow',
]

setup(
    name='loadflow',
    version='4.0.0',
    packages=packages,
    install_requires=install_requires,
)
