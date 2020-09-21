from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme_text = fh.read()

setup(
    name="dfpl",
    version="0.1",
    author="Jana Schor, Patrick Scheibe",
    author_email="jana.schor@ufz.de",
    packages=find_packages(),
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/yigbt/deepFPlearn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
