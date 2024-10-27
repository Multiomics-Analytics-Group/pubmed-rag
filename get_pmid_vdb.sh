#!/bin/bash

# Initialize variables
config="demo/config.yaml"
files_downloaded=None

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
	case $1 in
	-c | --config)
		config="$(readlink -f "$2")"
		shift
		;;
	-fd | --files-downloaded)
		files_downloaded="$(readlink -f "$2")"
		shift
		;;
	*)
		echo "Unknown parameter passed: $1"
		exit 1
		;;
	esac
	shift
done

# Print the parsed options
echo "Config file: $config"
echo "Files downloaded: $files_downloaded"

# Run get_embeddings.py and if success then run create_db.py
(
	cd "$(dirname "$(readlink -f "$0")")" &&
		python pubmed_rag/get_embeddings.py -c "$config" -fd "$files_downloaded" &&
		python pubmed_rag/create_db.py -c "$config"
)
