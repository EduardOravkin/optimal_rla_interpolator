import setuptools

setuptools.setup(
    name="orlai", 
    version="1.0.0",
    author="Eduard Oravkin",
    author_email="eduard.oravkin@gmail.com",
    description="Package which implements the Optimal Response-Linear Achievable Interpolator in Linear Regression.",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires = [
        'numpy==1.21.2',
        'scipy==1.7.0',
        'pandas==1.3.3',
        'scikit_learn==1.0.1',
    ]
)
