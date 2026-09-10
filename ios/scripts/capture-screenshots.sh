#!/bin/bash
# Grabs App Store screenshots from the booted simulator at the 6.5 inch size
# Apple requires (1284 x 2778).
#
# The iPhone 17 simulator renders 1206 x 2622, which is 0.5% narrower in aspect
# than Apple's slot. Scaling straight to 1284 x 2778 would stretch the image, so
# this scales on width and trims the surplus height instead.
#
#   ./capture-screenshots.sh out_dir name [name...]
#
# It pauses between shots so you can navigate the app to the next screen.

set -euo pipefail

W=1284
H=2778

out="${1:?usage: capture-screenshots.sh OUT_DIR NAME [NAME...]}"
shift
[ $# -gt 0 ] || { echo "give at least one screen name" >&2; exit 1; }

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
mkdir -p "$out"

xcrun simctl list devices booted | grep -q Booted || {
  echo "no booted simulator" >&2; exit 1
}

for name in "$@"; do
  read -r -p "Put the app on '$name', then press return: " _
  raw="$out/.$name.raw.png"
  xcrun simctl io booted screenshot --type=png "$raw" >/dev/null 2>&1

  # Scale on width, then trim the extra height from the centre.
  sips --resampleWidth "$W" "$raw" --out "$raw" >/dev/null
  sips -c "$H" "$W" "$raw" --out "$out/$name.png" >/dev/null
  rm -f "$raw"

  got=$(sips -g pixelWidth -g pixelHeight "$out/$name.png" | awk '/pixel/{printf "%s", $2 (NR==1?"x":"")}')
  echo "$name.png -> $got"
done

echo
echo "Upload these under Previews and Screenshots, iPhone 6.5 inch Display."
