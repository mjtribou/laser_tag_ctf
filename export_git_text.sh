#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./export_git_text.sh                    # text files only → ../<repo>-YYYYMMDD-HHMM-sources.txt
#   ./export_git_text.sh OUTFILE            # custom output path (text only)
#   ./export_git_text.sh OUTFILE all        # include non-text (hex or base64)

OUT="${1:-../$(basename "$(pwd)")-$(date +%Y%m%d-%H%M)-sources.txt}"
MODE="${2:-text}"   # 'text' (default) or 'all'

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repo." >&2
  exit 1
fi

# Collect files (tracked + untracked, excluding ignored), NUL-delimited → sorted list
paths_tmp="$(mktemp)"
git ls-files -z --cached --others --exclude-standard \
  | tr '\0' '\n' \
  | LC_ALL=C sort > "$paths_tmp"

if ! [ -s "$paths_tmp" ]; then
  echo "No files matched by git ls-files." >&2
  rm -f "$paths_tmp"
  exit 0
fi

# Gather metadata
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'no-branch')"
head_sha="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-commit')"
dirty=""
git diff --quiet || dirty="*DIRTY*"
git diff --cached --quiet || dirty="*DIRTY*"

# Start output
{
  echo "# Exported sources (git working tree)"
  echo "repo:    $repo_root"
  echo "branch:  $branch"
  echo "HEAD:    $head_sha $dirty"
  echo "date:    $(date -Is)"
  echo

  echo "=== TREE (only files selected by git ls-files) ==="
  awk -F'/' '
    # Use a local loop var (j) so we don’t clobber outer `i` (awk vars are global)
    function indent(n, j) { for (j = 0; j < n; j++) printf "  " }

    NF == 0 { next }               # skip blank lines
    { gsub(/^\.\/+/, "", $0) }     # normalize leading "./" if present

    {
      n = split($0, seg, "/")
      path = ""
      # Print unseen parent directories on first encounter
      for (i = 1; i < n; i++) {
        path = (path == "" ? seg[i] : path "/" seg[i])
        if (!(path in seen_dir)) {
          indent(i - 1); print seg[i] "/"
          seen_dir[path] = 1
        }
      }
      # Print the file under its directory
      indent(n - 1); print seg[n]
    }
  ' "$paths_tmp"

  echo
  echo "=== FILES (full contents) ==="
} > "$OUT"

# Helper: decide how to emit a file
emit_file() {
  local f="$1"
  local size sha mime
  size=$(wc -c < "$f" | tr -d ' ')
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  mime=$(file -b --mime-type "$f" 2>/dev/null || echo "application/octet-stream")

  {
    echo
    echo "----- BEGIN FILE: $f"
    echo "# size=${size}B  sha256=${sha}  mime=${mime}"
  } >> "$OUT"

  case "$MODE:$mime" in
    text:application/*|text:text/*) ;; # fallthrough handled below
  esac

  if [ "$MODE" = "all" ]; then
    # For binaries, prefer hex; else base64
    case "$mime" in
      text/*|application/json|application/xml|application/javascript|application/x-sh|application/x-httpd-php)
        cat "$f" >> "$OUT"
        ;;
      *)
        if command -v xxd >/dev/null 2>&1; then
          echo "# (hex dump)" >> "$OUT"
          xxd -g 1 "$f" >> "$OUT"
        else
          echo "# (base64)" >> "$OUT"
          base64 "$f" >> "$OUT"
        fi
        ;;
    esac
  else
    # text-only mode
    case "$mime" in
      text/*|application/json|application/xml|application/javascript|application/x-sh|application/x-httpd-php)
        cat "$f" >> "$OUT"
        ;;
      *)
        echo "[non-text omitted; run with 'all' to include]" >> "$OUT"
        ;;
    esac
  fi

  echo "----- END FILE: $f" >> "$OUT"
}

# Emit each file
while IFS= read -r f; do
  # Skip if it disappeared between listing and read
  [ -f "$f" ] || continue
  emit_file "$f"
done < "$paths_tmp"

rm -f "$paths_tmp"

echo "Wrote: $OUT"
