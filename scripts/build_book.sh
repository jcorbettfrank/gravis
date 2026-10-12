#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building WASM web-app ==="
(cd "$REPO_ROOT/crates/web-app" && wasm-pack build --target web --release)

echo "=== Building mdBook ==="
mdbook build "$REPO_ROOT/book"

echo "=== Copying WASM artifacts into book output ==="
BOOK_OUT="$REPO_ROOT/artifacts/book"
mkdir -p "$BOOK_OUT/demos/pkg"

cp "$REPO_ROOT/crates/web-app/pkg/web_app.js" "$BOOK_OUT/demos/pkg/"
cp "$REPO_ROOT/crates/web-app/pkg/web_app_bg.wasm" "$BOOK_OUT/demos/pkg/"
cp "$REPO_ROOT/crates/web-app/static/demos/"*.html "$BOOK_OUT/demos/"
cp "$REPO_ROOT/crates/web-app/static/demos/"*.js "$BOOK_OUT/demos/"
mkdir -p "$BOOK_OUT/demos/specs"
cp "$REPO_ROOT/specs/"*.json "$BOOK_OUT/demos/specs/"
bash "$REPO_ROOT/scripts/vendor_anthropic_sdk.sh" "$BOOK_OUT/demos/vendor/anthropic-sdk"
cp "$REPO_ROOT/crates/web-app/static/index.html" "$BOOK_OUT/demos/full.html"

echo "=== Done ==="
echo "Open $BOOK_OUT/index.html in a WebGPU-enabled browser"
