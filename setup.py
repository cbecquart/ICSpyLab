from setuptools import setup, find_packages
import pathlib

def main():
    # The directory containing this file
    HERE = pathlib.Path(__file__).parent

    # The text of the README file
    README = (HERE / "README.md").read_text()

    # Read dependencies from requirements.txt
    with open(HERE / "requirements.txt") as f:
        requirements = f.read().splitlines()

    setup(
        name='icspylab',
        version='0.1.0',
        author='Colombe Becquart, Abdallah Abdelsameia',
        author_email='colombe.becquart@tse-fr.eu, aabdelsameia1@gmail.com',
        maintainer="Colombe Becquart",
        maintainer_email="colombe.becquart@tse-fr.eu",
        description='Invariant Coordinate Selection (ICS) for multivariate data analysis.',
        long_description=README,
        long_description_content_type='text/markdown',
        url='https://github.com/cbecquart/ICSpyLab',
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.9',
    )

if __name__ == "__main__":
    main()
