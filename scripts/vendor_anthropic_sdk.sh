#!/bin/bash
# Copy the Anthropic TypeScript SDK's ES modules next to the text-to-universe
# page (crates/web-app/static/demos/universe.html), so a page that handles a
# visitor's API key loads no code from a CDN at run time (ADR 014).
# The version is pinned and the tarball is checked against its registry hash.
#
#   bash scripts/vendor_anthropic_sdk.sh <out-dir>   # writes <out-dir>/index.mjs, ...
set -euo pipefail

VERSION="0.130.0"
INTEGRITY="sha512-UMkwdFHvIdhYJHuPYkufmBTePKgk50RolnqZPUqyPclXp7LoVM7QgXsfQnx7KlmbkQtb+RhmUjN5ARYpbnz9YQ=="

OUT="$1"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

(cd "$TMP" && npm pack --silent "@anthropic-ai/sdk@$VERSION" >/dev/null)
TARBALL="$TMP/anthropic-ai-sdk-$VERSION.tgz"
ACTUAL="sha512-$(openssl dgst -sha512 -binary "$TARBALL" | openssl base64 -A)"
if [ "$ACTUAL" != "$INTEGRITY" ]; then
    echo "@anthropic-ai/sdk@$VERSION: integrity mismatch ($ACTUAL)" >&2
    exit 1
fi
tar xzf "$TARBALL" -C "$TMP"
mkdir -p "$OUT"
# Portable (macOS and Linux): no cp --parents
(cd "$TMP/package" && find . -name '*.mjs' | while read -r f; do
    mkdir -p "$OUT/$(dirname "$f")"
    cp "$f" "$OUT/$f"
done && cp LICENSE "$OUT/")
# The SDK's one bare import is the webhook-signature library, used only to
# verify webhooks server-side; the page maps it to this stub (an import map).
mkdir -p "$OUT/_stubs"
cat > "$OUT/_stubs/standardwebhooks.mjs" <<'STUB'
export class Webhook {
    constructor() {
        throw new Error('webhook verification is not available in this browser build');
    }
}
STUB
echo "Vendored @anthropic-ai/sdk@$VERSION -> $OUT"
