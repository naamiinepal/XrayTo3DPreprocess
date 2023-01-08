from setuptools import find_packages, setup

setup(name='xrayto3d_preprocess',
      version="0.1.0",
      description="Preprocessing utils and scripts for Dataset creation for X-ray to 3D Shape Reconstruction",
      url="https://github.com/naamiinepal/XrayTo3DPreprocess",
      author="Mahesh Shakya",
      author_email="mahesh.shakya@naamii.org.np",
      packages=find_packages(exclude=['build','docs']),
      install_requires=[
            'simpleitk<2.1.0', # temporary fix ITK ERROR: ITK only supports orthonormal direction cosines. No orthonormal definition found! 
            'omegaconf',
            'nibabel',
            'pandas'
      ],
      keywords="Python utilities simpleitk biomedical",
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Medical Science Apps.',

                   # Pick your license as you wish (should match "license" above)
                   'License :: OSI Approved :: MIT License',

                   # Specify the Python versions you support here. In particular, ensure
                   # that you indicate whether you support Python 2, Python 3 or both.
                   'Programming Language :: Python :: 3.6'
                   ])
