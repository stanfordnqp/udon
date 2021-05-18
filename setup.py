import setuptools

setuptools.setup(
    name='udon',
    version='0.1',
    description='Utilities for Differentiable Optics and Nanophotonics',
    author='Stanford Nanophotonics and Quantum Optics Lab',
    author_email='mr.jesselu@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "jax", "phidl", "scikit-image"],
    url='https://github.com/stanfordnqp/udon',
    classifiers=['License :: OSI Approved :: MIT License'],
)
