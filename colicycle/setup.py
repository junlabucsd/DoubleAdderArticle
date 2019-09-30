from setuptools import setup

setup(name='colicycle',
      version='0.1',
      description='Analyzing MoMA data',
      url='https://github.com/guiwitz/DoubleAdderArticle',
      author='Guillaume Witz',
      author_email='',
      license='MIT',
      packages=['colicycle'],
      zip_safe=False,
      install_requires=['numpy','scikit-image','scipy','jupyter','jupyterlab','pandas','h5py','tifffile', 'trackpy','tzlocal', 'ipympl', 'tabulate','simplegeneric','plotnine', 'xlrd','requests'],
      )