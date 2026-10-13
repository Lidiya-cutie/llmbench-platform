#!/bin/bash
cd /opt/llmbench
.venv/bin/python scripts/gen_dashboard_json.py >> /opt/llmbench/logs/dashboard_update.log 2>&1
