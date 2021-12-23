from setuptools import setup

install_requires = [
    'numpy'
]

packages = [
    'loadflow_lib',
]

setup(
    name='loadflow_lib',
    version='1.2.1',
    packages=packages,
    install_requires=install_requires,
)
