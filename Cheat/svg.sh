# CLI TOOLING FOR SVG
# ===================
# - imagemagick (the standard): HARD TO WORK WITH FOR SVG
#   - INSTALL:  sudo apt install imagemagick
#
# - inkscape (supreme godking of svg): require install entire app, can be overkill
#   - INSTALL:  sudo apt install inkscape
#
# - librsvg2-bin (`rsvg-convert`: JUUUUUUUST RIGHT!): best, easiest to use.
#   - INSTALL:  sudo apt install -y librsvg2-bin


# Convert an SVG to a favicon.ico (requires imagemagick).
convert -background transparent -define 'icon:auto-resize=16,24,32,64,128' logo.svg favicon.ico


#=========================  SVG CONVERSION & EXPORT  =========================#

# TOOL COMPARISON
# ===============
# imagemagick, inkscape, rsvg

#====  ImageMagick
convert -density 300 -background transparent -resize 1024x1024 raindropio.svg raindrop.png
# the true size shit is determined by the DPI val `-density`
#   - imagemagick is a real turd here and defaults to DPI for ANTS: 72
#   - and it is the DPI that determines image size, so with that
#     weak ass 72, you aint gettin SHIT from that
#   - so, you need to pickj a sufficiently large DPI such that you can then
#     RESIZE the image to the target dims


#====  Inkscape
inkscape raindropio.svg --export-type=png --export-width=1024 --export-height=1024 --export-background=transparent --export-filename=raindrop.png
# Yep, that's it. Simple.
# inkscape's whole dig IS svg after all.
# (fyi, you can also export to png in the gUI)


#====  librsvg2
rsvg-convert -f png -w 2048 -h 2048 -b transparent raindropio.svg > raindrop.png
# fucking done.
# ez
# how shit should be.
