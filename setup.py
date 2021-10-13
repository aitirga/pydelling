from setuptools import setup, find_packages

setup(
    name='PyFLOTRAN',  # How you named your package folder (MyLib)
    packages=find_packages(),  # Chose the same as "name"
    version='1.4',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Package to pre-process PFLOTRAN (and other software) files',
    # Give a short description about your library
    author='Aitor Iraola Galarza',  # Type in your name
    author_email='aitor.iraola@amphos21.com',  # Type in your E-Mail
    url='https://github.com/aitirga/PyFLOTRAN',  # Provide either the link to your github or to your website
    download_url='https://github.com/aitirga/PyFLOTRAN/archive/refs/tags/v_1.4.zip',  # I explain this later on
    keywords=['PFLOTRAN', 'Preprocessing', 'python', 'modelling'],  # Keywords that define your package best
    include_package_data=True,
    install_requires=[  # I get to this in a second
        'ofpp',
        'numpy',
        'pandas==1.0.5',
        'h5py==2.10.0',
        'vtk>=8.1.2',
        'matplotlib==3.3.0',
        'scipy==1.5.2',
        'python-box==5.1.0',
        'pyyaml==5.3.1',
        'setuptools~=51.0.0',
        'xlrd==1.2.0',
        'openpyxl',
        'tqdm',
        'seaborn',
        'open3D',
        'natsort',
        'pytecplot',
        'sklearn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
