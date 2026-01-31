# Awesome Zotero 📚️

[GH topics/zotero](https://github.com/topics/zotero)

# Installation & Setup

#### Zotero's official Linux distribution (bin)

📀 **Download here**: https://www.zotero.org/download/

🔧 **Instructions**: https://www.zotero.org/support/installation

> Download the tarball, extract the contents, and run `zotero` from that directory to start Zotero.
> 
> For Ubuntu, the tarball includes a .desktop file that can be used to add Zotero to the launcher:
> 
> 1. Move the extracted directory to a location of your choice (e.g., `/opt/zotero`).
> 
> 2. Run the `set_launcher_icon` script from a terminal to update the .desktop file for that location. .desktop files require absolute paths for icons, so `set_launcher_icon` replaces the icon path with the current location of the icon based on where you've placed the directory.
> 
> 3. Symlink `zotero.desktop` into `~/.local/share/applications/` (e.g., `ln -s /opt/zotero/zotero.desktop ~/.local/share/applications/zotero.desktop`)
> 
> Zotero should then appear either in your launcher or in the applications list when you click the grid icon (“Show Applications”), from which you can drag it to the launcher.
> 
> You may need to re-run `set_launcher_icon` after certain Zotero updates. If something isn't working, it may help to remove the current symlink (`~/.local/share/applications/zotero.desktop`), wait a few seconds for Zotero to disappear from the launcher, and recreate it.

#### Flatpak (preferred)

**Flathub page**: https://flathub.org/apps/org.zotero.Zotero

Installation:

```
flatpak install flathub org.zotero.Zotero
```

(Add filesystem permission for Zotero):

```
sudo flatpak override --filesystem=home org.zotero.Zotero
```

#### Deb

- **Package src**: [retorquere/zotero-deb: Packaged versions of Zotero for Debian-based systems](https://github.com/retorquere/zotero-deb)

> *To install Zotero, use the following commands:*

```
wget -qO- https://raw.githubusercontent.com/retorquere/zotero-deb/master/install.sh | sudo bash
sudo apt update
sudo apt install zotero
```

You can also just grab the deb from the host: https://mirror.mwt.me/zotero/deb/



----

# Plugins

- Plugin Template: [windingwind/zotero-plugin-template: A plugin template for Zotero.](https://github.com/windingwind/zotero-plugin-template)

- [MuiseDestiny/zotero-gpt: GPT Meet Zotero.](https://github.com/MuiseDestiny/zotero-gpt) 🔥

- [l0o0/tara: A Zotero add-on for backup and restore preferences, add-ons, translators, styles, and locate between two machines](https://github.com/l0o0/tara)

- [Creling/Zotero-Metadata-Scraper: A Zotero plugin that automatically retrieves and updates paper metadata from multiple academic sources based on paper titles.](https://github.com/Creling/Zotero-Metadata-Scraper)

- [ethanwillis/zotero-scihub: A plugin that will automatically download PDFs of zotero items from sci-hub](https://github.com/ethanwillis/zotero-scihub)

- [scitedotai/scite-zotero-plugin: scite zotero plugin](https://github.com/scitedotai/scite-zotero-plugin)

- [sobamchan/zotero_tldr_api: Generate summaries for papers in your Zotero library](https://github.com/sobamchan/zotero_tldr_api)

- [Dominic-DallOsto/zotero-pin-items: Pin items in your collections in Zotero so they always appear at the top.](https://github.com/Dominic-DallOsto/zotero-pin-items)

- 

### File Management

- Zotfile

- [ChenglongMa/zoplicate: A plugin that does one thing only: Detect and manage duplicate items in Zotero.](https://github.com/ChenglongMa/zoplicate)

- 

### Notes

- [windingwind/zotero-better-notes: Everything about note management. All in Zotero.](https://github.com/windingwind/zotero-better-notes)

- [daeh/zotero-markdb-connect: Zotero plugin that links your Markdown database to Zotero. Jump directly from Zotero Items to connected Markdown files. Automatically tags Zotero Items so you can easily see which papers you&#39;ve made notes for.](https://github.com/daeh/zotero-markdb-connect)

- fdsa

## Themes & Customization

- [tefkah/zotero-night: Night theme for Zotero UI and PDF](https://github.com/tefkah/zotero-night)

- 

---

# Clients

### Android
Zotero now has an *OFFICIAL* android app:

`org.zotero.android`:
- Big Brother Store: https://play.google.com/store/apps/details?id=org.zotero.android
- Source (no builds): https://github.com/zotero/zotero-android



---

# Community

## Swole Doges

#### windingwind (Xiangyu Wang)

- Page: https://windingwind.github.io/
- GH: [windingwind](https://github.com/windingwind)

##### Works

- [**windingwind/zotero-plugin-template**: A plugin template for Zotero.](https://github.com/windingwind/zotero-plugin-template)
   - [**windingwind/zotero-plugin-toolkit**: Toolkit for Zotero Plugin Developers.](https://github.com/windingwind/zotero-plugin-toolkit)
- 🔥 [**<mark>windingwind/zotero-better-notes</mark>**: Everything about note management. All in Zotero.](https://github.com/windingwind/zotero-better-notes)
- [**windingwind/zotero-pdf-translate**: Translate PDF, EPub, webpage, metadata, annotations, notes to the target language. Support 20+ translate services.](https://github.com/windingwind/zotero-pdf-translate)
- [**windingwind/zotero-actions-tags**: Customize your Zotero workflow.](https://github.com/windingwind/zotero-actions-tags)

#### MuiseDestiny

[MuiseDestiny (Polygon)](https://github.com/MuiseDestiny)

##### Works

- [MuiseDestiny/zotero-reference: PDF references add-on for Zotero.](https://github.com/MuiseDestiny/zotero-reference)
- [MuiseDestiny/zotero-citation: Make Zotero&#39;s citation in Word easier and clearer.](https://github.com/MuiseDestiny/zotero-citation)
- [MuiseDestiny/zotero-gpt: GPT Meet Zotero.](https://github.com/MuiseDestiny/zotero-gpt)
- [MuiseDestiny/zotero-attanger: Attachment Manager for Zotero](https://github.com/MuiseDestiny/zotero-attanger)
   - **THIS IS ALLEGEDLY SOMETHING LIKE A "ZOTFILE" FOR Zotero 7 (zotfile only supports Zotero 6)**
- 

#### Retorquere

- [retorquere (Emiliano Heyns)](https://github.com/retorquere)
   - [Better BibTeX for Zotero](https://retorque.re/zotero-better-bibtex/)
      - [retorquere/zotero-better-bibtex: Make Zotero effective for us LaTeX holdouts](https://github.com/retorquere/zotero-better-bibtex)
   - [retorquere/zotero-date-from-last-modified](https://github.com/retorquere/zotero-date-from-last-modified)

----

Other cool plugins:

- [GitHub - volatile-static/Chartero: Chart in Zotero](https://github.com/volatile-static/Chartero)
- 
