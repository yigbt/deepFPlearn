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
        "jsonpickle",
        "pandas",
        "rdkit-pypi",
        "keras",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        'console_scripts': ['dfpl=dfpl.__main__:main']
    }
)