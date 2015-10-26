#!/bin/bash


if [ ! -d "$1" ]; then
	echo "Error"
	exit 1
fi

if [ -f "$2" ]; then
	if [ -f "$2" ]; then
		rm -f "$2"
	else
		echo "File '$2' is not writable"
		exit 1
	fi
fi

find "$1" -iname "*.jp*" -or -iname "*.pn*" > __temp && \
wc -l __temp | awk '{ print $1 }' > "$2" && \
cat __temp >> "$2" && \
rm -f __temp
