from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme_text = fh.read()

setup(
    name="dfpl",
    version="0.1",
    author="Jana Schor, Patrick Scheibe, Matthias Bernt",
    author_email="jana.schor@ufz.de",
    packages=find_packages(include=['dfpl', 'dfpl.*']),
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/yigbt/deepFPlearn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=[
        "jsonpickle~=2.1",
        "matplotlib==3.5.1",
        "numpy==1.19.5",
        "pandas==1.4.2",
        "rdkit-pypi==2022.03.1",
        "scikit-learn==1.0.2",
        "tensorflow=2.6.0",
        "wandb~=0.12",
    ],
    entry_points={
        'console_scripts': ['dfpl=dfpl.__main__:main']
    }
)