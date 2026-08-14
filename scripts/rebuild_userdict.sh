#!/usr/bin/env bash
# Rebuild the SBV2 user dictionary (all.bin) with lindera 1.4.1 so it matches
# jpreprocess 0.13.2's lindera. The all.bin distributed on HuggingFace
# (neody/sbv2-api-assets) is compiled with a mismatched lindera version, so its
# word features decode to garbage (dropped/wrong Japanese readings). Rebuilding
# the SAME AivisSpeech source with the matching lindera fixes it.
#
# This replaces upstream scripts/make_dict.sh (which uses `lindera build -k ipadic`
# = the broken version). Dictionary DATA is unchanged; we only normalize junk
# field encodings (cform 0/empty -> *, which means "no conjugation" for these
# noun entries; readings are untouched).
#
#   Usage: scripts/rebuild_userdict.sh [OUT]   (default OUT=$HOME/.cache/sbv2/all.bin,
#          where sbv2_core/build.rs looks — so it skips the HF download.)
#
# Requires: git, cargo (to build mkuserdict), network. See doc
# kakuyomu-tts/doc/06_sbv2_allbin_lindera_mismatch.md for the full write-up.
set -euo pipefail

OUT="${1:-$HOME/.cache/sbv2/all.bin}"
AIVIS_COMMIT="168b2a1144afe300b0490d9a6dd773ec6e927667"   # AivisSpeech-Engine dict source (pinned)
JP_REPO="https://github.com/kazu/jpreprocess.git"
JP_BRANCH="tool/mkuserdict"                                # mkuserdict = CSV -> lindera-1.4.1 .bin

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[rebuild_userdict] 1/4 build mkuserdict (lindera 1.4.1) from $JP_BRANCH"
git clone --depth 1 --branch "$JP_BRANCH" "$JP_REPO" "$WORK/jp"
export CARGO_TARGET_DIR="$WORK/jp/target"
( cd "$WORK/jp" && cargo build --release -p jpreprocess-dictionary --bin mkuserdict )
MKU="$WORK/jp/target/release/mkuserdict"

echo "[rebuild_userdict] 2/4 fetch AivisSpeech dictionary CSVs @ $AIVIS_COMMIT"
git clone --filter=blob:none -n https://github.com/Aivis-Project/AivisSpeech-Engine "$WORK/aivis"
( cd "$WORK/aivis" && git checkout "$AIVIS_COMMIT" -- 'resources/dictionaries/*.csv' )

echo "[rebuild_userdict] 3/4 concat 0*.csv, pad to 16 cols, normalize junk fields"
: > "$WORK/all.csv"
for f in "$WORK"/aivis/resources/dictionaries/0*.csv; do
  cat "$f" >> "$WORK/all.csv"
  echo >> "$WORK/all.csv"
done
# 全行16列にパディング(Metadata::default() は strict CSV) + cform(col10) 0/空→*, ctype(col9) 空/**→*
awk -F',' 'BEGIN{OFS=","}
  {
    while (NF < 16) $(NF+1) = ""
    if ($10 == "0" || $10 == "") $10 = "*"
    if ($9  == ""  || $9  == "**") $9 = "*"
    print
  }' "$WORK/all.csv" > "$WORK/all16n.csv"

echo "[rebuild_userdict] 4/4 compile -> $OUT"
mkdir -p "$(dirname "$OUT")"
"$MKU" "$WORK/all16n.csv" "$OUT"
echo "[rebuild_userdict] done: $(wc -c < "$OUT") bytes -> $OUT"
