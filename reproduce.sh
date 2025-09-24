#!/bin/bash

find configs -type f ! -name ".*" -not -path "*/snippets/*" | while read -r file; do
    python main "$file"
done
