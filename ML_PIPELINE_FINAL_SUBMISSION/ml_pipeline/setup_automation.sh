#!/bin/bash
# =============================================================================
# setup_automation.sh
# Sets up automated daily pipeline runs
# =============================================================================

echo "🤖 Setting up ML Pipeline Automation"
echo ""

# Get the full path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Create log directory for cron
mkdir -p "$PROJECT_DIR/logs/cron"

# Create the cron job command
CRON_CMD="0 2 * * * cd $PROJECT_DIR && /usr/bin/python3 auto_pipeline.py --source api >> logs/cron/automation.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "auto_pipeline.py"; then
    echo "⚠️  Cron job already exists. Skipping..."
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job added!"
    echo ""
    echo "Pipeline will run daily at 2:00 AM"
fi

echo ""
echo "Current cron jobs:"
crontab -l

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ AUTOMATION SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "What happens now:"
echo "  ✓ Every day at 2:00 AM, the pipeline will:"
echo "    1. Fetch live data from your API/database"
echo "    2. Run full pipeline (clean, validate, train)"
echo "    3. Auto-deploy if model performance is good"
echo "    4. Email you ONLY if something fails"
echo ""
echo "To test immediately:"
echo "  python auto_pipeline.py --source api"
echo ""
echo "To remove automation:"
echo "  crontab -e  (then delete the line with auto_pipeline.py)"
echo ""
