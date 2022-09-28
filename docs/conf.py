# Configuration file for the Sphinx documentation builder.

# -- Project information

project = '人工智能药物设计教程'
author = '2022, Abdusemi Abduweli'
copyright ='😜'
release = ''
version = ''

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = "furo"


# -- Options for EPUB output
epub_show_urls = 'footnote'
