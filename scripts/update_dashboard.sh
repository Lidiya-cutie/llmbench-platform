#!/bin/bash
cd /opt/llm-data/llmbench
.venv/bin/python scripts/gen_dashboard_json.py >> /opt/llm-data/llmbench/logs/dashboard_update.log 2>&1
