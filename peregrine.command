#!/bin/sh
# Double-click this in Finder to run the Peregrine wizard in a Terminal
# window. It keeps the window open at the end so you can read the output.
# From a terminal, run ./peregrine instead.

"$(dirname "$0")/peregrine" "$@"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "  Wizard exited with code $rc."
fi
echo
printf "  Press Return to close this window. "
read -r _
exit "$rc"
