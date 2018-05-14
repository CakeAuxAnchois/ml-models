from setuptools import setup

setup(name='ml-models',
      description='Python implementation of machine learning models',
      long_description=open("README").read(),
      url='https://github.com/CakeAuxAnchois/ml-models',
      author='Roselyn Titon',
      license='MIT',
      packages=['models'],
      install_requires=[
          'numpy',
          'cvxopt'
      ],
      zip_safe=False)
