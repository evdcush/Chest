# *Operator Protocols*
Sounds cooler than "Notes on How to Do/Operate Stuff". 😎

```{tableofcontents}
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

