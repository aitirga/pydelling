from setuptools import setup, find_packages

setup(
    name='pydelling',  # How you named your package folder (MyLib)
    packages=find_packages(),  # Chose the same as "name"
    version='1.6.3',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Package to pre-process and help on mathematical modelling',  # Give a short description about your library
    author='Aitor Iraola Galarza',  # Type in your name
    author_email='aitirga@gmail.com',  # Type in your E-Mail
    url='https://github.com/aitirga/pydelling',  # Provide either the link to your github or to your website
    download_url='https://github.com/aitirga/pydelling/archive/refs/tags/v_1.6.3.zip',  # I explain this later on
    keywords=['PFLOTRAN', 'Preprocessing', 'python', 'modelling', 'postprocessing', 'math'],  # Keywords that define your package best
    include_package_data=True,
    install_requires=[  # I get to this in a second
        'ofpp',
        'numpy',
        'pandas',
        'h5py',
        'vtk',
        'matplotlib',
        'scipy',
        'python-box',
        'pyyaml',
        'setuptools',
        'xlrd',
        'openpyxl',
        'tqdm',
        'seaborn',
        'open3D',
        'natsort',
        'pytecplot',
        'sklearn',
        'munch',
        'meshio',
        'plotly',
        'shapely',
        'tabulate',
        'mkdocs',
        'mkdocs-material',
        'mkdocstrings',
        'streamlit',
        'dill'
        'sympy'
        'trimesh'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
