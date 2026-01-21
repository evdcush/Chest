#!/usr/bin/env bash
# md2pdf.sh
#     self-contained Markdown --> PDF
#
# REQ: python3-markdown, weasyprint
#
# Usage: ./md2pdf.sh file.md

#set -euo pipefail
echo 'fuck0'
#=============================================================================#
#                             1. argument checking                            #
#=============================================================================#
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 file.md" >&2
  exit 1
fi
echo 'fuck2'

INPUT="$1"
[[ -f "$INPUT" ]] || { echo "File not found: $INPUT" >&2; exit 1; }
echo 'fuck3'

#=============================================================================#
#                                   2. paths                                  #
#=============================================================================#
BASENAME="${INPUT%.md}"
PDF="${BASENAME}.pdf"
echo 'fuck4'
#=============================================================================#
#                                3. inline CSS                                #
#=============================================================================#
read -r -d '' CSS <<'CSS_BLOCK'
/* A4, 15 mm margins, light theme ----------------------------------------- */
@page {
  size: A4;
  margin: 10mm;
}
html, body {
  margin: 0;
  padding: 0;
  font-family: "Noto Serif", "Noto Serif CJK JP", serif;
  font-size: 12pt;
  line-height: 1.5;
  color: #000;
  background: #fff;
}
h1, h2, h3, h4, h5, h6 {
  margin-top: .6em;
  margin-bottom: .6em;
  border-bottom: 1px solid #e1e4e8;
}
pre, code {
  font-family: "Noto Sans Mono", monospace;
  background: #f6f8fa;
}
pre {
  padding: .6em .8em;
  overflow: auto;
}
table { border-collapse: collapse; }
th, td {
  border: 1px solid #d0d7de;
  padding: 4px 6px;
}
CSS_BLOCK

echo 'fuck5'

#=============================================================================#
#           4. Markdown → PDF (single step via process substitution)          #
#=============================================================================#
weasyprint <(
  cat <<EOF
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>${BASENAME}</title>
  <style>
${CSS}
  </style>
</head>
<body>
$(python -m markdown -x tables -x fenced_code "$INPUT")
</body>
</html>
EOF
) "$PDF"

echo 'fuck7'

echo "Generated: $PDF"
