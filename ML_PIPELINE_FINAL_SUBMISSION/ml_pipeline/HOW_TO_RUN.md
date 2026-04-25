# 🎓 HOW TO RUN THIS PROJECT (Explained Like You're 5)

## 📦 STEP 1: Download and Extract

### What to do:
1. **Download** the file `ML_PIPELINE_FINAL_SUBMISSION.zip`
2. **Right-click** on the ZIP file
3. Click **"Extract All..."** (Windows) or **"Unzip"** (Mac)
4. Choose where to save it (like your Desktop or Downloads folder)
5. You should now have a folder called `ml_pipeline`

---

## 🔧 STEP 2: Install Python (If You Don't Have It)

### Check if you have Python:
1. Press `Windows Key + R` (Windows) or `Cmd + Space` (Mac)
2. Type: `cmd` (Windows) or `terminal` (Mac)
3. Press Enter
4. Type: `python --version`
5. Press Enter

**If you see "Python 3.8" or higher** → You're good! Skip to Step 3.

**If you see an error** → Download Python from https://www.python.org/downloads/
- Click the big yellow "Download Python" button
- Run the installer
- ✅ **IMPORTANT:** Check the box that says "Add Python to PATH"
- Click "Install Now"

---

## 💻 STEP 3: Open the Project in VS Code

### Install VS Code (if you don't have it):
1. Go to https://code.visualstudio.com/
2. Click "Download"
3. Install it (just click Next, Next, Next, Install)

### Open the project:
1. **Open VS Code**
2. Click **"File"** at the top
3. Click **"Open Folder..."**
4. Find the `ml_pipeline` folder you extracted
5. Click **"Select Folder"**

You should now see all the files on the left side of VS Code!

---

## 📚 STEP 4: Install the Required Libraries

Think of this like downloading all the toys you need to play with.

### In VS Code:
1. Look at the top menu
2. Click **"Terminal"**
3. Click **"New Terminal"**

A window should open at the bottom of VS Code. This is called the **terminal**.

### Type these commands (press Enter after each one):

#### For Windows:
```bash
pip install -r requirements.txt
```

#### For Mac/Linux:
```bash
pip3 install -r requirements.txt
```

**What's happening?** 
The computer is downloading all the tools the pipeline needs (pandas, scikit-learn, etc.)

**This takes 1-2 minutes.** You'll see lots of text scrolling. That's normal!

When it's done, you'll see "Successfully installed..." and the cursor will be blinking again.

---

## ▶️ STEP 5: Run the Pipeline!

### The moment of truth:

In that same terminal at the bottom of VS Code, type:

```bash
python main.py
```

Press **Enter**.

### What you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║          ML PIPELINE — BINARY CLASSIFICATION                ║
║          Production-Style Data Pipeline                     ║
╚══════════════════════════════════════════════════════════════╝
```

Then it will run through 5 stages:
- ✅ Stage 1: INGESTION (loading data)
- ✅ Stage 2: CLEANING (fixing data)
- ✅ Stage 3: VALIDATION (checking quality)
- ✅ Stage 4: DECISION (deciding whether to train)
- ✅ Stage 5: TRAINING (training 4 models)

**This takes about 1-2 minutes.**

At the end, you'll see:

```
✅ PIPELINE COMPLETED SUCCESSFULLY
Model Metrics:
  Accuracy  : 0.7792
  ROC-AUC   : 0.8287
```

🎉 **Congratulations! The pipeline ran successfully!**

---

## 🎨 STEP 6: View the Dashboard

### After the pipeline finishes:

1. In VS Code, look at the left sidebar (file explorer)
2. Click the **triangle** next to `models` to expand it
3. Find **`dashboard.html`**
4. **Right-click** on `dashboard.html`
5. Choose **"Reveal in File Explorer"** (Windows) or **"Reveal in Finder"** (Mac)
6. **Double-click** `dashboard.html`

Your web browser will open and show you a beautiful interactive dashboard with:
- Model performance charts
- SHAP feature importance plot
- Data quality metrics

---

## 🎥 STEP 7: Recording Your Video

### Setup:

1. **Close unnecessary apps** (Discord, Spotify, etc.)
2. **Open these in order:**
   - Your GitHub repo in Chrome/Edge
   - VS Code with the project
   - Terminal ready
3. **Have dashboard.html ready to open**

### Recording Options:

#### **Option 1: OBS Studio (Free, Best Quality)**
1. Download from https://obsproject.com/
2. Install it
3. Open OBS
4. Click **"Start Recording"**
5. Do your demo
6. Click **"Stop Recording"**
7. Videos save to: `C:\Users\YourName\Videos` (Windows) or `~/Movies` (Mac)

#### **Option 2: Windows Built-in (Easiest)**
1. Press `Windows Key + G`
2. Click the **record button** (circle)
3. Do your demo
4. Press `Windows Key + G` again
5. Click **stop**
6. Videos save to: `C:\Users\YourName\Videos\Captures`

#### **Option 3: Mac Built-in**
1. Press `Cmd + Shift + 5`
2. Choose **"Record Entire Screen"**
3. Click **Record**
4. Do your demo
5. Click **Stop** in the menu bar

---

## 📤 STEP 8: Upload Video and Submit

### Upload to Google Drive:
1. Go to https://drive.google.com
2. Click **"New"** → **"File upload"**
3. Choose your video
4. Wait for upload to finish
5. **Right-click** on the video
6. Click **"Get link"**
7. Change to **"Anyone with the link"**
8. Click **"Copy link"**

**OR**

### Upload to YouTube (Unlisted):
1. Go to https://youtube.com
2. Click your profile picture → **"YouTube Studio"**
3. Click **"Create"** → **"Upload videos"**
4. Choose your video
5. Set visibility to **"Unlisted"**
6. Copy the link

---

## 📧 STEP 9: Send the Email

### Email Template:

**To:** moh@scriptchain.co

**Subject:** ML Internship Assignment — Mokshit — Binary Classification Pipeline

**Body:**

```
Hi Moh,

Please find my ML internship assignment submission:

GitHub Repository:
[PASTE YOUR GITHUB LINK HERE]

Video Walkthrough (5 min):
[PASTE YOUR VIDEO LINK HERE]

Key Highlights:
• Production-grade ML pipeline for binary classification
• Multi-model training (LR, RF, XGBoost, LightGBM) with auto-selection
• SHAP explainability for interpretable predictions
• Real Pima Indians Diabetes dataset (82.9% ROC-AUC)
• 3-layer validation + statistical drift detection
• Interactive HTML dashboard with model comparison
• Bonus: Fully automated version (auto_pipeline.py) for production deployment

The system runs out-of-the-box with:
  pip install -r requirements.txt
  python main.py

Looking forward to discussing this further.

Best regards,
Mokshit
```

---

## 🆘 TROUBLESHOOTING

### "pip is not recognized"
**Fix:** 
```bash
python -m pip install -r requirements.txt
```

### "No module named pandas"
**Fix:**
```bash
pip install pandas scikit-learn xgboost lightgbm shap matplotlib seaborn
```

### "Permission denied"
**Fix (Windows):** Run VS Code as Administrator
**Fix (Mac/Linux):**
```bash
pip3 install --user -r requirements.txt
```

### Pipeline runs but no dashboard appears
**Fix:**
1. Go to the `models` folder
2. Find `dashboard.html`
3. Double-click it manually

### Video is too large to upload
**Fix:**
- Compress it using https://www.freeconvert.com/video-compressor
- Or use YouTube instead of Google Drive

---

## ✅ FINAL CHECKLIST

Before you submit:

- [ ] Pipeline runs successfully (`python main.py`)
- [ ] Dashboard opens in browser
- [ ] GitHub repo is public
- [ ] Video is 5 minutes or less
- [ ] Video link works (test it in incognito mode!)
- [ ] Email has both links (GitHub + video)
- [ ] Submit BEFORE 11:00am ET April 25, 2026

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Install everything
pip install -r requirements.txt

# Run the pipeline
python main.py

# Run for specific date
python main.py --date 2026-04-24

# Run without training (validation only)
python main.py --dry-run

# Get help
python main.py --help
```

---

## 📞 If Something Goes Wrong

**Don't panic!** Common issues have simple fixes above.

If you're truly stuck:
1. Take a screenshot of the error
2. Google the error message
3. Check the troubleshooting section above

**You've got this! 🚀**
