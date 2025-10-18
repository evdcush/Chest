# Processing Codebases

Resources for ingesting a codebase.

## Code structure and understanding

- [tree-sitter: Tree-sitter is a parser generator tool and an incremental parsing library. It can build a concrete syntax tree for a source file and efficiently update the syntax tree as the source file is edited](https://github.com/tree-sitter/tree-sitter)
  - Docs: https://tree-sitter.github.io/tree-sitter/
  - Python bindings: https://github.com/tree-sitter/py-tree-sitter
    - docs: https://tree-sitter.github.io/py-tree-sitter/
    - install: `pip install tree-sitter`

- [**ahmedkhaleel2004/gitdiagram** - Turn any GitHub repository into an interactive diagram for visualization in seconds. Replace `hub` with `diagram` for any GH url to access its diagram.](https://github.com/ahmedkhaleel2004/gitdiagram)
  - Site: https://gitdiagram.com/


## Repo to File

- [**yamadashy/repomix** - Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file.](https://github.com/yamadashy/repomix)
  - **Handy site:** https://repomix.com/
  - **Browser ext.**:
    - FF: https://addons.mozilla.org/firefox/addon/repomix/
    - BigBrother: https://chromewebstore.google.com/detail/repomix/fimfamikepjgchehkohedilpdigcpkoa
  - **vsc ext**: https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner

- [**cyclotruc/gitingest** - Replace 'hub' with 'ingest' in any GitHub URL to get a prompt-friendly extract of a codebase](https://github.com/cyclotruc/gitingest)
  - *apparently now [coderamp-labs/gitingest](https://github.com/coderamp-labs/gitingest)*
  - **Handy site**: https://gitingest.com/
  - **CLI**: `pip install gitingest # or 'gitingest[server]'`
  - **Browser ext**:
    - FF: https://addons.mozilla.org/firefox/addon/gitingest
    - BigBrother: https://chromewebstore.google.com/detail/adfjahbijlkjfoicpjkhjicpjpjfaood

- [**mufeedvh/code2prompt** - CLI tool to convert your codebase into a single LLM prompt with source tree, prompt templating, and token counting](https://github.com/mufeedvh/code2prompt)
  - Docs (kinda?): https://code2prompt.dev/
  - install via ... cargo: `cargo install code2prompt`

- [🔥🔥 **kleneway/pastemax** - A simple tool to select files from a repository to copy/paste into an LLM. **DESKTOP APP**](https://github.com/kleneway/pastemax)
  - **GUI APP!**: https://github.com/kleneway/pastemax/releases/download/v1.1.0-stable/pastemax_1.1.0-stable_amd64.deb


- [**kamilstanuch/codebase-digest** - Codebase-digest is your AI-friendly codebase packer and analyzer. Features 60+ coding prompts and generates structured overviews with metrics. Ideal for feeding](https://github.com/kamilstanuch/codebase-digest)
  - install: `pip install codebase-digest`

-----


### Repomix
> [yamadashy/repomix - Repomix is a powerful tool that packs your entire repository into a single, AI-friendly file.](https://github.com/yamadashy/repomix)

**Handy site**: https://repomix.com/


#### installation
CLI:
```sh
npm i -g repomix
```


##### USAGE

**NB**: *output in different formats with the `--style` flag, eg `--style markdown`, `plain`, `json`, etc.*


**BASICS**
```sh
# pack your repo
repomix

# pack a specific repo
repomix path/to/dir

# pack specific files/dirs globbed:
repomix --include "src/**/*.ts,**/*.md"

# exclude:
repomix --ignore "**/*.log,tmp/"
```

**REMOTE**
```sh
# pack remote repo
repomix --remote https://github.com/yamadashy/repomix

# can use gh shorthand:
repomix --remote yamadashy/repomix

# specify the branch name, tag, or commit hash:
repomix \
--remote https://github.com/yamadashy/repomix \
--remote-branch main

# specific commit hash:
repomix \
--remote https://github.com/yamadashy/repomix \
--remote-branch 935b695

# specifying the branch's URL
repomix --remote https://github.com/yamadashy/repomix/tree/main

# commit url
repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

**FILE LIST (stdin pipe)**
```sh
# Using find command
find src -name "*.ts" -type f | repomix --stdin

# Using git to get tracked files
git ls-files "*.ts" | repomix --stdin

# Using grep to find files containing specific content
grep -l "TODO" **/*.ts | repomix --stdin

# Using ripgrep to find files with specific content
rg -l "TODO|FIXME" --type ts | repomix --stdin

# Using ripgrep (rg) to find files
rg --files --type ts | repomix --stdin

# Using sharkdp/fd to find files
fd -e ts | repomix --stdin

# Using fzf to select from all files
fzf -m | repomix --stdin

# Interactive file selection with fzf
find . -name "*.ts" -type f | fzf -m | repomix --stdin

# Using ls with glob patterns
ls src/**/*.ts | repomix --stdin

# From a file containing file paths
cat file-list.txt | repomix --stdin

# Direct input with echo
echo -e "src/index.ts\nsrc/utils.ts" | repomix --stdin
```





----

## Other converters of `X` to text

- [**unclecode/crawlai** - 🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler & Scraper.](https://github.com/unclecode/crawl4ai)
  - Docs: https://docs.crawl4ai.com/

- [**firecrawl/firecrawl** - 🔥 The Web Data API for AI. Turn entire websites into LLM-ready markdown or structured data](https://github.com/firecrawl/firecrawl)


- [**mishushakov/llm-scraper** - Turn any webpage into structured data using LLMs](https://github.com/mishushakov/llm-scraper)

