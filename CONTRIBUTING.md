# 🤝 Contributing guidelines

Contributions are welcome and greatly appreciated! 
Whether you're fixing a bug, improving documentation, or proposing a new feature, your help is valuable.

---

## Contributing via issues 


### Reporting bugs

If you encounter a bug, please open an issue on [GitHub](https://github.com/cbecquart/ICSpyLab/issues) and include:

- A clear and descriptive title  
- Steps to reproduce the issue  
- Expected vs. actual behavior  
- Python version and operating system  
- Relevant logs or error messages  

Before submitting, please check if the issue has already been reported.


### Suggesting enhancements

We welcome ideas for new features or improvements.

To propose an enhancement:
- Open an issue describing your idea  
- Explain the use case and potential impact  
- (Optional) Suggest an implementation approach  

For significant changes, please discuss them first before submitting a pull request.

---

## Contributing via pull requests

If you want to contribute to the code of ICSpyLab, the best way is to 
fork the repository on GitHub and submit a pull request. 


### Development setup

To set up the project locally, fork and clone the ICSpyLab repository. 

```bash
git clone https://github.com/cbecquart/ICSpyLab.git
cd ICSpyLab
```

Create a branch with a relevant name, for instance:
- `feature/...` for new features
- `fix/...` for bug fixes
- `docs/...` for documentation changes

Then, you need to set up a local development environment. 
We strongly recommend to create a separate virtual environment containing all dependencies.
To do so, use the ``environment.yml`` file.

```bash
conda env create --file environment.yml
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Development guidelines

- All changes related to one feature must belong to one branch.
- All code must follow the standard Python guidelines for code style, PEP8.
- Each function, class, method, and attribute needs to be documented using doc strings.
- Ensure your code is compatible with the License. 
- If your contribution introduces a new method: include reference papers and examples when possible.
- If your contribution changes functionality, please update the documentation accordingly.


### Tests

New features and bug fixes should include appropriate tests, using pytest. 
To run the tests locally:
```bash
pytest
```

All tests must be part of the ``tests/`` folder and follow the existing structure. 
Please ensure that all tests pass before submitting a pull request.


### Documentation

The documentation resides in the ``docs/`` folder and is written in reStructuredText (rst), or Markdown (md) for examples.
HTML files of the documentation can be generated using Sphinx.

On Windows:
```bash
cd docs
./make.bat clean html
./make.bat html
```

On Linux/macOS:
```bash
cd docs
make clean html
make html
```

Generated files will be located in ``docs/_build/html``.


### Pull request process
1. Fork the repository
2. Create a new branch
3. Make your changes with clear commit messages 
4. Add or update tests if needed
5. Update the documentation if needed
6. Ensure all tests pass and code is properly formatted
7. Open a [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) with a clear description

Please keep pull requests focused and concise. If your PR addresses an issue, link it in the description.

---

## Thank you

Thank you for taking the time to contribute!
Your support helps improve the project for everyone 🐍
