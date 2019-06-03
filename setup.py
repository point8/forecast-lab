from distutils.core import setup

setup(
    name='datascience101',
    version=0.1,
    description='',
    author='Christian Staudt',
    url='http://www.point-8.de',
    packages=['forecast_lab'],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "fbprophet",
        "statsmodels"
    ]
)
