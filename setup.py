### import required functions 
from setuptools import setup,find_packages

### do the set up 
setup(
	author='R.  Austin Benn',
	description='package to run CCA analysis using the HCP dataset',
	name='CCAtools',
	version='0.1.0',
	packages=find_packages(include=['CCAtools','CCAtools.*']),
	install_requires=['numpy','nibabel','matplotlib','scikit-learn','seaborn','networkx','surfdist','scipy','nilearn'],
	python_requires='>=3.4'
	)