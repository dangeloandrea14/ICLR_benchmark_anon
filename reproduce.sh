#!/bin/bash

find configs -type f ! -name ".*" -not -path "*/snippets/*" | while read -r file; do
    echo python main "$file"
done
