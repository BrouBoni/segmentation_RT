from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='segmentation_RT',
    version='0.0.1a1',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(include=["segmentation_rt", "segmentation_rt.*"]),
    url='https://github.com/BrouBoni/segmentation_RT',
    license='MIT',
    author='Kévin N. D. Brou Boni & Julien Laffarguette',
    author_email='k-brouboni@o-lambret.fr',
    description='Python library for radiotherapy deep learning segmentation',
    python_requires='>=3.8',

    classifiers=[

            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.8',
            'Operating System :: Unix',
        ],

    keywords='radiotherapy, segmentation, deep learning',

    install_requires=['scikit_image>=0.18.1',
                      'matplotlib>=3.3.4',
                      'natsort>=7.1.1',
                      'dcmrtstruct2nii>=1.0.19',
                      'scipy>=1.6.0',
                      'torchio>=0.18.29',
                      'nibabel>=3.2.1',
                      'pydicom>=2.1.2',
                      'torch>=1.7.1',
                      'tensorboard>=2.4.1',
                      'numpy>=1.19.2',
                      'pypng>=0.0.20',
                      ],

    zip_safe=False,
)
