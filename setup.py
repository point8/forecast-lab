from distutils.core import setup

setup(
    name='forecast-lab',
    version=0.1,
    description='',
    author='Christian Staudt',
    url='https://point-8.de',
    packages=['forecast_lab'],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "fbprophet",
        "statsmodels"
    ]
)
