# CardioSense - Screen Recording Demo Script

**Total Duration:** ~8-10 minutes  
**Setup:** Have the terminal and file explorer open; ensure .venv is activated.

---

## **SEGMENT 1: INTRODUCTION (1 min)**

### **SPEAK:**
"Hi! I'm showing you CardioSense, a data science project that predicts heart disease risk using probability and statistics. The project has three main parts: data analysis, a web app for exploration, and interactive visualizations. Let me walk you through how it works."

### **SHOW:**
- Open file explorer and show project structure (folders: src/, heart.csv, requirements.txt, README.md)
- Brief screenshot of GitHub repo (show README)

---

## **SEGMENT 2: CODE OVERVIEW (1.5 min)**

### **SPEAK:**
"The project is built in Python with three key modules. First, there's the analysis module that loads data, computes statistics, confidence intervals, and trains a logistic regression model. Second, the Streamlit app gives an interactive dashboard. And the dataset—heart.csv with 14 variables and a target column for disease presence or absence."

### **SHOW:**
- Open `src/analysis.py` in editor
- Scroll through and highlight key functions:
  - `load_data()`
  - `confidence_intervals()`
  - `probability_summary()`
  - `train_logistic_regression()`
  - `generate_plots()`
- Open `src/app.py` and show the Streamlit imports and `render_header()` function
- Briefly show `heart.csv` columns

---

## **SEGMENT 3: INSTALLATION (1 min)**

### **SPEAK:**
"To run this project, first we install the dependencies. The project uses pandas, numpy, scikit-learn for analysis, and streamlit for the web app."

### **SHOW & RUN:**
```bash
# Make sure you're in the project directory
cd ~/Desktop/All/Probability\ Project

# Activate virtual environment (if not already done)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Display output:** Watch pip install packages. Takes ~30 seconds.

---

## **SEGMENT 4: RUN ANALYSIS (2 min)**

### **SPEAK:**
"Now let's run the analysis. This script loads the heart dataset, computes descriptive statistics, calculates 95% confidence intervals for numeric features, computes probability summaries—like the likelihood of disease given sex and binomial probabilities—then trains a logistic regression model and generates visualizations."

### **SHOW & RUN:**
```bash
python src/analysis.py
```

**SPEAK (while running):**
"This is processing the data, training the model, and generating plots. You'll see output showing the dataset summary, confidence intervals, and model metrics like accuracy, precision, recall, F1 score, and AUC."

### **SHOW:**
- Display the printed output (dataset summary, CIs, model metrics)
- Show the outputs that are generated if available

**SPEAK:**
"The analysis computes key insights about the data. Notice the model performance metrics—AUC shows how well it can distinguish between disease and no disease."

---

## **SEGMENT 5: LAUNCH STREAMLIT APP (3 min)**

### **SPEAK:**
"Next, let's launch the interactive web app. This is a Streamlit dashboard where you can explore the data visually and make predictions. It pulls data from our analysis module and caches results for speed."

### **SHOW & RUN:**
```bash
streamlit run src/app.py
```

**SPEAK (while Streamlit loads):**
"Streamlit is starting a local server. In a few seconds, the app will open in your browser at localhost:8501."

### **SHOW:**
Once the app opens, walk through each section:

**1) Hero Section (30 sec):**
- Point to the title "Heart Disease Risk Prediction"
- Show the pills: Records count, number of variables, target rate
- SPEAK: "This header summarizes the dataset at a glance. We have 303 records with 14 features."

**2) Dataset Overview Tab (1 min):**
- Click "Dataset Overview" tab
- Point to stat cards (rows, columns, missing values)
- Scroll down to the data preview table
- SPEAK: "Here you see summary statistics and a preview of the first 20 rows. You can identify missing values and get a feel for the raw data. Notice we have features like age, sex, cholesterol, and maximum heart rate achieved."

**3) Descriptive Statistics Tab (30 sec):**
- Click "Descriptive Statistics"
- Show the summary table (mean, std, min, max, etc.)
- Show the confidence intervals table
- SPEAK: "These tables show the range and uncertainty around key numeric features. The confidence intervals tell us where the true population mean likely falls. For example, the average age is around 54 with confidence bounds."

**4) Probability & Distribution Tab (30 sec):**
- Click "Probability & Distribution"
- Show the cards with probability values (target rate, conditional on sex, binomial sample, cholesterol > 240)
- SPEAK: "These are probability summaries computed from the data. For example, we show the overall disease rate and the rate split by sex, which helps identify at-risk groups. About 54% of patients in this dataset have heart disease."

**5) Model & Prediction Tab (1 min):**
- Click "Model & Prediction"
- Show the model metrics (accuracy, precision, recall, F1, AUC)
- Show the ROC curve plot
- Scroll to the prediction form; fill in sample values:
  - age: 55
  - sex: 1 (male)
  - chol: 240
  - thalach: 150
- Click Predict
- SPEAK: "The model takes your inputs and gives a risk probability. An output near 1.0 means high disease risk; near 0.0 means low risk. This model achieves 85%+ accuracy and helps provide early screening aid."

### **SPEAK (summary):**
"The dashboard is interactive, responsive, and lets anyone—even without coding knowledge—explore the data and run predictions. The metrics show our logistic regression model performs very well with high accuracy and AUC."

---

## **SEGMENT 6: SUMMARY & CLOSING (1 min)**

### **SPEAK:**
"Let me recap what we saw. CardioSense integrates data science workflows: loading and analyzing data, training a predictive model, visualizing results, and launching an interactive web app—all with clean, modular Python code. The project demonstrates probability, statistics, machine learning, and web development in one real-world application. You can install it locally, run the analysis, launch the dashboard, and make predictions. The code is well-structured with separate modules for analysis and the app, making it easy to understand and extend. Thanks for watching!"

### **SHOW:**
- Quick screenshot of the GitHub repo link
- Final view of the project folder structure

---

## **OPTIONAL ADDITIONS:**

### **If time allows:**
- Show the `requirements.txt` and explain each dependency
- Show `README.md` in the browser/editor
- Open `.gitignore` and explain version control best practices
- Show git log: `git log --oneline` (to highlight commits)
- Mention: "In production, we'd add unit tests for each module and deploy this to Heroku or AWS for public access"

### **Technical Notes:**
- Keep terminal visible throughout for credibility
- Use zoom/enlarge text if recording on high-res screen
- Pause for ~2 seconds after each major action so viewers can read
- Mute notification sounds before recording
- Test all commands beforehand to avoid errors on camera
- Record at 1080p or higher for clarity
- Use a clear microphone for audio quality

---

## **SPEAKING TIPS:**
- Speak clearly and moderately paced
- Pause between segments for viewer comprehension
- Use phrases like "Notice how…", "Here you can see…", "This demonstrates…"
- Emphasize the value: "This saves time", "Easy to use", "Practical application"
- End on a high note about what's learned and next steps
- Be enthusiastic but professional

---

## **EDITING (Post-Recording):**
- Trim intro/outro silence
- Add captions for key terms (logistic regression, AUC, ROC curve)
- Add text overlays with file paths or command syntax for clarity
- Background music is optional (keep it subtle, royalty-free)
- Total video length: ~8-10 minutes
