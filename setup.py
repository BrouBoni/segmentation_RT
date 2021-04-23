from setuptools import setup

setup(
    name='segmentation_RT',
    version='0.0.1a1',
    packages=['dl', 'dl.model', 'dl.dataloader', 'util', 'mask2rs', 'rs2mask'],
    package_dir={'': 'segmentation_rt'},
    url='https://github.com/BrouBoni/segmentation_RT',
    license='MIT',
    author='Kévin N. D. Brou Boni & Julien Laffarguette',
    author_email='k-brouboni@o-lambret.fr',
    description='Python library for radiotherapy deep learning segmentation',

    classifiers=[

            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.8',
            'Operating System :: Unix',
        ],

    keywords='Radiotherapy, segmentation, deep learning',

    install_requires=['scikit_image==0.18.1',
                      'matplotlib==3.3.4',
                      'natsort==7.1.1',
                      'dcmrtstruct2nii==1.0.19',
                      'scipy==1.6.0',
                      'torchio==0.18.29',
                      'nibabel==3.2.1',
                      'pydicom==2.1.2',
                      'torch==1.7.1',
                      'numpy==1.19.2',
                      'Pillow==8.1.2',
                      'pypng==0.0.20'],
)
