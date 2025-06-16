# Contributing

In general, we welcome contributions from the community, including bug fixes and documentation improvements. Please see the Issues tab for discussion.

## Development environment

To get started, clone the repository and set up your local environment:

```bash
# Clone the repository
git clone https://github.com/pynapple-org/pynapple.git
cd pynapple

# Create a new conda environment
conda create --name pynapple pip python=3.8
conda activate pynapple

# Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"
```

Note: If you're an external contributor, you'll likely want to fork the repository first with your own GitHub account, and then set up an `upstream` remote branch:

```
# Replace username with your GitHub username
git clone https://github.com/<username>/pynapple.git
git remote add upstream https://github.com/pynapple-org/pynapple
```

## Git workflow

In general, we recommend developing changes on feature branches, and then opening a pull request against the `dev` branch for review.

```bash
# Create a new branch
git checkout dev
git pull origin dev # or git pull upstream dev
git checkout -b your-branch-name

# Commit and push changes
git commit -m "Your commit message"
git push origin your-branch-name
```

## Testing and linting

You can run the tests and code linters locally using `tox`. This is generally kept in sync with the Github Actions defined in `.github/workflows` via the [`tox.ini`](tox.ini) file.

```bash
# Install tox
pip install tox tox-conda

# Run tests and linter
tox
```

## Generating docs

The user documentation is generated using Sphinx and can be built using the Makefile in the `doc/` folder:

```bash
cd doc && make html
```

You can also start a development server that will watch for changes with `sphinx-autobuild`:

```bash
sphinx-autobuild . _build/html
```
