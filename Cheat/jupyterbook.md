# Jupyter Book

Notes on working with Jupyter Book.

# Workflow

## Install

```
pip install --upgrade jupyter-book
```

Overview of commands:

```
$ jupyter-book --help
Usage: jupyter-book [OPTIONS] COMMAND [ARGS]...

  Build and manage books with Jupyter.

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  build   Convert your book's or page's content to HTML or a PDF.
  clean   Empty the _build directory except jupyter_cache.
  config  Inspect your _config.yml file.
  create  Create a Jupyter Book template that you can customize.
  myst    Manipulate MyST markdown files.
  toc     Command-line for sphinx-external-toc.
```

## create

```
jupyter-book create docs
```

This creates a jupyter-book template `docs` in your current directory.

```
$ ls docs
references.bib
notebooks.ipynb
intro.md
markdown.md
markdown-notebooks.md
logo.png
requirements.txt
_config.yml
_toc.yml
```

# Build and Publish to GH Pages

Use `ghp-import`:

```
pip install --upgrade ghp-import
```
