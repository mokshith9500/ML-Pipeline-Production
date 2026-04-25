# 🤖 AUTOMATION QUICK START GUIDE

## Turn Your Pipeline Into a Zero-Touch System

This guide shows you how to set up **fully automated** pipeline runs with live data.

---

## What You Get

Once set up, the system will:

- ✅ **Fetch live data** from your API/database/S3 every day at 2am
- ✅ **Run the full pipeline** automatically (ingest → clean → validate → train)
- ✅ **Auto-deploy** the best model if performance meets thresholds
- ✅ **Email you** ONLY when something fails (no noise!)
- ✅ **Zero manual intervention** required

---

## Setup (Choose Your Platform)

### **Option 1: Linux/Mac (Cron)**

```bash
# 1. Make the setup script executable
chmod +x setup_automation.sh

# 2. Run the setup
./setup_automation.sh

# 3. Edit config.yaml with your data source credentials
nano config.yaml  # Update live_data section

# Done! Pipeline runs daily at 2am automatically
```

### **Option 2: Windows (Task Scheduler)**

```powershell
# 1. Open PowerShell as Administrator

# 2. Run the setup script
.\setup_automation_windows.ps1

# 3. Edit config.yaml with your data source credentials
notepad config.yaml  # Update live_data section

# Done! Pipeline runs daily at 2am automatically
```

---

## Configuration (IMPORTANT)

Edit `config.yaml` and add your live data source:

### For API Data Source

```yaml
live_data:
  api:
    url: "https://api.yourcompany.com/patient-data"
    auth_token: "your_token_here"
    params:
      days: 1
```

### For Database

```yaml
live_data:
  database:
    type: "postgresql"
    host: "db.yourcompany.com"
    port: 5432
    database: "patient_records"
    user: "ml_user"
    password: "your_password"
    query: "SELECT * FROM records WHERE date = CURRENT_DATE"
```

### For AWS S3

```yaml
live_data:
  s3:
    bucket: "your-bucket"
    prefix: "daily-data/"
    aws_access_key: "your_key"
    aws_secret_key: "your_secret"
```

### Email Alerts

```yaml
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "pipeline@yourcompany.com"
    password: "your_app_password"  # Gmail: use app-specific password
    to_emails:
      - "you@company.com"
```

---

## Testing Before Automation

**Test the automation BEFORE scheduling it:**

```bash
# Test API data fetch
python auto_pipeline.py --source api

# Test database fetch
python auto_pipeline.py --source database

# Test S3 fetch
python auto_pipeline.py --source s3
```

If it runs successfully, the automation will work!

---

## How It Works

### Daily Workflow (Runs at 2am Every Day)

```
1. Fetch live data from your configured source
   ↓
2. Save to data/raw/live_data_YYYY-MM-DD.csv
   ↓
3. Run pipeline: Clean → Validate → Train
   ↓
4. Check validation status
   ├─ PASS → Continue to training
   └─ FAIL → Email alert + stop
   ↓
5. Train 4 models, select best
   ↓
6. Check model performance
   ├─ ROC-AUC >= 0.80 AND F1 >= 0.65 → Auto-deploy + email success
   └─ Below threshold → Skip deployment
   ↓
7. Log everything to logs/cron/automation.log
```

### You Get Emailed When:

- ❌ Data fetch fails
- ❌ Pipeline crashes
- ❌ Validation fails
- ✅ New model deployed successfully

**You DON'T get emailed when:**
- ✅ Everything works fine (no noise!)
- ⏸️ Model trained but not deployed (performance below threshold)

---

## Monitoring

### Check Automation Logs

```bash
# Linux/Mac
tail -f logs/cron/automation.log

# Windows
Get-Content logs\cron\automation.log -Wait
```

### Check Deployed Models

```bash
cat models/deployment_history.json
```

### Check Metrics Over Time

```bash
cat models/metrics_registry.json
```

---

## Advanced: Customizing the Schedule

### Linux/Mac (Cron)

```bash
# Edit crontab
crontab -e

# Examples:
# Every day at 3am:     0 3 * * * ...
# Twice daily (6am, 6pm): 0 6,18 * * * ...
# Every hour:           0 * * * * ...
# Every Monday at 9am:  0 9 * * 1 ...
```

### Windows (Task Scheduler)

1. Open Task Scheduler (`taskschd.msc`)
2. Find "ML_Pipeline_Daily_Run"
3. Right-click → Properties → Triggers
4. Edit the schedule as needed

---

## Stopping Automation

### Linux/Mac

```bash
crontab -e
# Delete the line with auto_pipeline.py
```

### Windows

```powershell
Unregister-ScheduledTask -TaskName "ML_Pipeline_Daily_Run"
```

---

## For ScriptChain's Use Case

**Example configuration for patient health monitoring:**

```yaml
live_data:
  database:
    type: "postgresql"
    host: "scriptchain-db.internal"
    database: "patient_health"
    query: |
      SELECT 
        patient_id,
        glucose_level,
        bmi,
        blood_pressure_systolic,
        blood_pressure_diastolic,
        metabolic_syndrome_risk as target
      FROM daily_patient_readings
      WHERE reading_date = CURRENT_DATE
      AND data_validated = true

alerts:
  email:
    to_emails:
      - "ml-ops@scriptchain.co"
      - "data-team@scriptchain.co"

deployment:
  enabled: true
  production_model_path: "models/production/metabolic_risk_model.pkl"
  min_roc_auc: 0.82  # Higher threshold for medical predictions
```

---

## Troubleshooting

### "Authentication failed"
- Check your API token / database password in config.yaml
- For Gmail: use an app-specific password, not your regular password

### "cron job not running"
```bash
# Check if cron service is running
sudo systemctl status cron  # Linux
sudo service cron status    # Mac
```

### "Task not running on Windows"
- Run PowerShell as Administrator when setting up
- Check Task Scheduler logs: Event Viewer → Task Scheduler → History

### "Email not sending"
- Gmail users: Enable "Less secure app access" OR use app-specific password
- Check SMTP server and port are correct

---

## What This Demonstrates to ScriptChain

This automation setup shows you understand:

✅ **Production MLOps** — Not just training models, but deploying them  
✅ **Reliability** — Email alerts, retry logic, error handling  
✅ **Scalability** — Works with APIs, databases, cloud storage  
✅ **Zero-touch operation** — Runs without human intervention  
✅ **Real-world deployment** — Auto-deployment based on performance thresholds  

This is exactly what a production ML system at ScriptChain would need.
