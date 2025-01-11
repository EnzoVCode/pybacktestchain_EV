from setuptools import setup, find_packages

setup(
    name='pybacktestchain-ev',
    version='0.1.2',
    author='Enzo Volpato',
    author_email='enzo.volpato@outlook.fr',
    description='Improved pybacktestchain library with interactive UI and features added (risk measures, trading strategies, multiple asset classes)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/EnzoVCode/python_project',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'pytest',
        'yfinance',
        'sec-cik-mapper',
        'streamlit',
    ],
)
