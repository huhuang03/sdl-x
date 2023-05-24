from setuptools import setup, find_packages

packages = find_packages()
packages.remove('samples')
packages.remove('test')

setup(
    name='sdl',
    version='1.0.0',
    description='收集的简单deep learning方法',
    author="th",
    packages=packages,
    install_requires=['numpy']
)
