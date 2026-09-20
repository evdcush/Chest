# *Operator Protocols*
Sounds cooler than "Notes on How to Do/Operate Stuff". 😎

```{tableofcontents}
```

# jupyter book

## `sphinx-book-theme`
src: https://github.com/executablebooks/sphinx-book-theme

Here are the knobs you get for the theme (**i think**), from
`sphinx-book-theme/src/sphinx_book_theme/theme/sphinx_book_theme/theme.conf`:

```
[theme]
inherit = pydata_sphinx_theme
pygments_style = tango
sidebars = navbar-logo.html, icon-links.html, search-button-field.html, sbt-sidebar-nav.html
stylesheet = styles/sphinx-book-theme.css

[options]
# Announcement bar (empty by default)
announcement =

# Secondary sidebar: Removes the extra page-specific links from sidebar
secondary_sidebar_items = page-toc.html
toc_title = Contents

# Article header: The top bar that is displayed on most pages
article_header_start = toggle-primary-sidebar.html
article_header_end = article-header-buttons.html
use_download_button = True
use_fullscreen_button = True
use_issues_button = False
use_source_button = False
use_repository_button = False
# Note: We also inherit use_edit_page_button from the PyData theme

# Configuration for GitHub / GitLab repo buttons
path_to_docs =
repository_url =
repository_branch =
repository_provider =
launch_buttons = {}

# Header / navbar.
# Over-ride the PST navbar components (users could provide their own if they wish)
navbar_start =
navbar_center =
navbar_end =
navbar_persistent =

# Primary sidebar behavior
home_page_in_toc = False
show_navbar_depth = 1
max_navbar_depth = 4
collapse_navbar = False

# Footer at the bottom of the content
extra_footer =
footer_content_items = author.html, copyright.html, last-updated.html, extra-footer.html

# Footer at the bottom of the site (PST over-ride)
footer_start =
footer_end =

# Content and directive flags
use_sidenotes = False

# DEPRECATE after a few release cycles
expand_toc_sections = []

```

Of potential interest:
- `use_sidenotes` ?
- the sidebar behavior stuff (`home_page_in_toc`, `max_navbar_depth`, etc.)
- `use_download_button = True` <-- you probably want this false?
- "we also inherit ... from PyData theme": **many more knobs here maybe**
  - `use_edit_page_button`

#### in their styles
`sphinx-book-theme/src/sphinx_book_theme/assets/styles/`

`base/_base.scss`:

**PRINT CONTROL SETTINGS**:
```scss
/**
 * Printing behavior
 */
// Only display upon printing
.onlyprint {
  display: none;

  @media print {
    display: block !important;
  }
}

// Prevent an item from being printed
.noprint {
  @media print {
    display: none !important;
  }
}

```

### PyData theme stuff

#### FONT

`bt src/pydata_sphinx_theme/assets/styles/variables/_fonts.scss`

```css
html {
  /*****************************************************************************
  * Font features used in this theme
  */

  // base font size - applied at body/html level
  --pst-font-size-base: 1rem;

  // heading font sizes based on a medium contrast type scale
  // - see: https://github.com/Quansight-Labs/czi-scientific-python-mgmt/issues/97#issuecomment-2310531483
  --pst-font-size-h1: 2.625rem;
  --pst-font-size-h2: 2.125rem;
  --pst-font-size-h3: 1.75rem;
  --pst-font-size-h4: 1.5rem;
  --pst-font-size-h5: 1.25rem;
  --pst-font-size-h6: 1rem;

  // smaller than heading font sizes
  --pst-font-size-milli: 0.9rem;

  // Sidebar styles
  --pst-sidebar-font-size: 0.9rem;
  --pst-sidebar-font-size-mobile: 1.1rem;
  --pst-sidebar-header-font-size: 1.2rem;
  --pst-sidebar-header-font-weight: 600;

  // Admonition styles
  --pst-admonition-font-weight-heading: 600;

  // Font weights
  --pst-font-weight-caption: 300;
  --pst-font-weight-heading: 600;

  // Font family
  // https://modernfontstacks.com/?stack=system-ui
  --pst-font-family-base-system:
    system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji",
    "Segoe UI Symbol", "Noto Color Emoji";

  // https://modernfontstacks.com/?stack=monospace-code
  --pst-font-family-monospace-system:
    ui-monospace, "Cascadia Code", "Source Code Pro", "Menlo", "Consolas",
    "DejaVu Sans Mono", monospace;
  --pst-font-family-base: var(--pst-font-family-base-system);
  --pst-font-family-heading: var(--pst-font-family-base-system);
  --pst-font-family-monospace: var(--pst-font-family-monospace-system);
}

$line-height-body: 1.65;
$fa-font-path: "vendor/fontawesome/webfonts/";
```


# Misc


## MyST-editor
A live-preview myst editor.

Installation:

```sh
# clone the src
git clone --depth=1 https://github.com/antmicro/myst-editor && \
cd myst-editor

# install
npm i
## fix the vulnerabilities of the deps used:
npm audit fix

# build the pkg
npm run build
```

Launch that puppy:

```sh
npm run host
# open whatver the localhost shows, eg
http://localhost:5173/
```

