import setuptools
setuptools.setup(
    name='pyolo',
    version='0.0',
    scripts=['./scripts/myscript'],
    author='Pedro Fontana',
    description='YOLO v2 implementation in Keras (with TensorFlow backend)',
    packages=['lib.myscript']
    install_requires=[
        'setuptools',
        'tensorflow==2.2.0',
        'PyYAML==5.4.1',
        'numpy==1.20.2',
        'matplotlib==3.4.1'
    ],
    python_requires='>=3.8'
)