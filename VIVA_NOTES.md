# Viva notes (every member should be able to answer all of these)

**What problem does the model solve?** It predicts whether an online session ends in a purchase. Binary classification (purchase / no purchase).

**What do the main features mean?**
- `Administrative`, `Informational`, `ProductRelated` (+ `_Duration`): number of pages of each type visited and seconds spent on them.
- `BounceRates`: share of visitors who leave after one page. `ExitRates`: share of views of a page that were the last in a session.
- `PageValues`: average value of pages visited before a transaction.
- `SpecialDay`: closeness of the visit to a special day (e.g. Valentine's Day), 0 to 1.
- `Month`, `VisitorType`, `Weekend`: when and who. `OperatingSystems`, `Browser`, `Region`, `TrafficType`: integer codes for categories.

**Why this preprocessing?** Numerical columns are on very different scales (seconds in the thousands, rates below 1), so they are standardised; this matters for KNN and Logistic Regression. Text columns need encoding, and the integer-coded columns are one-hot encoded too because the codes are labels, not quantities. Everything sits inside a Pipeline so it is fitted only on training data.

**Why is the data imbalanced a problem?** About 85% of sessions are "no purchase", so a model that always says "no purchase" gets ~85% accuracy and is useless. That is why we use stratified splits, class weights, and judge by F1, recall, precision and ROC-AUC of the purchase class.

**Why stratify?** So the train, test and every CV fold keep the same ~15% purchase share.

**Why compare these four algorithms?** Logistic Regression is the simple linear baseline; KNN is distance-based; Decision Tree is interpretable but overfits; Random Forest averages many trees to reduce overfitting. Comparing train F1 with CV F1 shows who overfits.

**Metrics.** Precision: of sessions predicted as purchases, how many really were. Recall: of real purchases, how many we caught. F1: balance of the two. ROC-AUC: how well the model ranks buyers above non-buyers across all thresholds (0.5 = chance, 1 = perfect). Confusion matrix: counts of TN, FP, FN, TP.

**Which matters more, precision or recall?** Depends on the use. If the action is cheap (a popup), recall matters more. If it is costly (a large discount), precision matters more. State your choice and justify it.

**What is cv=5?** The training data is split into 5 folds; the model trains 5 times, each fold validating once. Average scores are more reliable than one split.

**What did GridSearchCV do?** Tried every combination in the chosen model's parameter grid, scored each with 5-fold CV on F1, and kept the best (the grid, the best values and the number of fits are in `reports/metrics.json`).

**Why a Pipeline?** Same preprocessing in training, CV, testing and the app, and no leakage: inside CV the scaler and encoder are fitted on each training fold only.

**What is in model.pkl?** The whole fitted Pipeline: fitted scaler + fitted encoder + fitted tuned classifier.

**How does Streamlit send input to the model?** Widget values are placed into a one-row DataFrame with the training column order and passed to `model.predict_proba(row)`.

**Why is training not repeated on Predict?** Training and tuning are slow and one-off; the app loads the saved pipeline once (cached) and only calls `predict_proba`, which takes milliseconds.

**What is the model score?** The model's confidence for "purchase". With class weighting it is not a true probability.

**Why was `PageValues` a concern?** It is derived from purchase-related activity, so it can leak the outcome; check how much performance drops without it.

**How was Git used? Local vs deployed?** Commits per task from each member. Deployed app runs the same code and `model.pkl` on a cloud server from the GitHub repo and installs packages from `requirements.txt`.

**Limitations.** One retailer and year, imbalanced target, unseen months, undocumented category codes, educational only.
