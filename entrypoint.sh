#!/bin/sh
# Fix /data ownership then drop to botuser
mkdir -p /data
chown -R botuser:botuser /data
exec su-exec botuser python main.py
