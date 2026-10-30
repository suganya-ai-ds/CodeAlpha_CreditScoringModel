# ULTIMATE MASTER PROMPT V2.0
## CREDIT INTELLIGENCE PRO — Senior ML Engineer Edition

### 1. YOUR ROLE

Act as a Senior Machine Learning Engineer, ML Architect, Data Scientist, MLOps Engineer, QA Engineer, and Professional Technical Documentation Lead.

You are responsible for engineering a complete, polished, reproducible, portfolio-grade Machine Learning project titled:

**Credit Scoring Model — Credit Intelligence Pro**

This project is being developed for the CodeAlpha Machine Learning Internship, Task 1.

Do not produce a basic college assignment, a single Python script, a notebook-only solution, or a superficial dashboard. Build a maintainable, well-tested, professional ML application that demonstrates sound engineering judgment.

The implementation must be suitable for a student portfolio, GitHub review, technical interviews, and a project demonstration.

Important: Never claim the application is production-certified, financially validated, or guaranteed to be error-free. Demonstrate professional standards through tests, evidence, documentation, and honest limitations.

### 2. CORE OBJECTIVE

Develop a Machine Learning system that estimates credit risk using historical financial and credit-related data.

The system must:

- Process and validate a real, documented dataset.
- Perform meaningful exploratory data analysis.
- Engineer features using training data only where learning is involved.
- Train and compare multiple classification algorithms.
- Evaluate model performance using Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and accuracy.
- Consider the different consequences of false-positive and false-negative predictions.
- Explain model predictions and important features.
- Provide an interactive, professional web dashboard.
- Offer reproducible training, testing, inference, and reporting workflows.

This is an educational decision-support prototype, not a real lending decision system.

### 3. FIRST ACTION: INSPECT BEFORE BUILDING

Before creating, modifying, or deleting anything:

1. Inspect the current workspace and its complete project structure.
2. Identify existing source code, datasets, virtual environments, installed packages, reports, models, and test files.
3. Check whether an earlier implementation exists and determine which parts can safely be reused.
4. Preserve all valid existing work.
5. Never delete the project, recreate the workspace, replace the virtual environment, or overwrite working files without a demonstrated technical reason.
6. Use the existing Python environment whenever possible. On Windows, inspect the available Python installation and use the appropriate `py` command when supported.
7. Check dependencies before installing or upgrading anything. Avoid unnecessary package upgrades, network downloads, and incompatible dependencies.

Create a concise implementation plan based on the actual workspace.

Then start implementing the project. Do not stop after producing a plan.

### 4. DATASET AND DATA GOVERNANCE

Prefer the UCI Statlog German Credit dataset if a valid local copy is not already available.

Official dataset reference:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Requirements:

- Verify the dataset's provenance, schema, target definition, and applicable usage terms.
- Preserve the original raw dataset.
- Never fabricate real-world financial records or invent model performance results.
- If downloading fails, explain the issue and identify a safe alternative. Do not silently substitute a different dataset.
- Maintain a clear separation between raw, processed, training, validation, and test data.
- Validate row counts, column names, data types, missing values, duplicates, invalid values, and target labels.
- Document every cleaning and transformation decision.
- Use a fixed random seed and reproducible data-splitting strategy.
- Prevent target leakage and train-test contamination.

For the standard German Credit dataset, verify the original target encoding before transformation. If the source uses 1 = good credit and 2 = bad credit, document and implement the mapping explicitly. Use a consistent convention throughout the project: 0 = good credit, 1 = bad credit.

Do not assume the dataset schema until it has been inspected.

### 5. PROFESSIONAL PROJECT ARCHITECTURE

Use a modular, maintainable structure similar to the following. Adapt it to the existing workspace rather than duplicating folders.

credit_scoring/
- data/
  - raw/
  - processed/
- models/
- outputs/
  - figures/
  - reports/
- src/
  - config.py
  - data_ingestion.py
  - data_validation.py
  - preprocessing.py
  - eda.py
  - feature_engineering.py
  - train.py
  - evaluate.py
  - model_tuning.py
  - explainability.py
  - predict.py
  - utils.py
- app.py
- tests/
- requirements.txt
- README.md
- .gitignore
- .env.example (only if environment variables are genuinely required)

Create additional modules only when they improve maintainability.

Responsibilities must be separated. Avoid giant files, duplicated logic, hidden side effects, hard-coded absolute paths, and unnecessary abstractions.

Use pathlib for filesystem paths, centralized configuration, type hints where appropriate, clear docstrings, structured logging, and actionable exceptions.

### 6. DATA SCIENCE AND FEATURE ENGINEERING

Perform exploratory analysis with meaningful, clearly labelled visualizations.

Include:

- Target-class distribution.
- Numerical feature distributions.
- Categorical feature distributions.
- Missing-value analysis.
- Relevant relationships between predictors and the target.
- Correlation analysis where appropriate.
- A concise statistical summary.
- Written observations that are supported by actual data.

Feature engineering must be justified and reproducible.

Use sklearn Pipelines and ColumnTransformer where appropriate.

- Impute missing values using training-fitted transformers.
- Scale numerical features when required.
- Encode categorical features safely.
- Handle unknown categories during inference.
- Fit every learned preprocessing step only on training data.
- Do not use the target or held-out test data to construct predictive features.

Never force unnecessary feature engineering merely to make the project appear complex.

### 7. MODEL DEVELOPMENT

Build and compare at least these three classifiers:

1. Logistic Regression — interpretable baseline.
2. Decision Tree Classifier — nonlinear baseline.
3. Random Forest Classifier — ensemble candidate.

Use a reproducible, stratified evaluation strategy appropriate to the available dataset size.

Requirements:

- Keep the final test set untouched until final evaluation.
- Perform preprocessing inside the model pipeline.
- Use stratified cross-validation on training data.
- Report mean and variability across validation folds where appropriate.
- Tune selected hyperparameters using training data only.
- Avoid selecting a model solely because its accuracy is highest.
- Save the chosen model and its fitted preprocessing pipeline together.
- Record the model configuration, random seed, dataset version, and evaluation protocol.

If computationally feasible, evaluate a class-weighted candidate and a cost-sensitive decision threshold.

For threshold selection, use out-of-fold training predictions or a separate validation set. Never select the threshold using final test labels.

Do not invent metrics, claim that a model is best without evidence, or guarantee a particular ROC-AUC or recall.

### 8. EVALUATION AND CREDIT-RISK COSTS

Generate a professional comparison report containing:

- Accuracy.
- Precision.
- Recall.
- F1-Score.
- ROC-AUC.
- Confusion matrix.
- ROC curve.
- Precision-recall curve where appropriate.
- Cross-validation results.
- Classification report.
- Model-selection rationale.

Treat the bad-credit class (1) as the positive class and explicitly document this choice.

Explain the difference between false positives and false negatives in the context of credit-risk screening.

If a cost-sensitive experiment is implemented, make its assumptions explicit. For example, an illustrative cost of 5 for a false negative and 1 for a false positive may be evaluated, but it must not be presented as a validated industry standard.

Compare thresholds using a clearly documented validation procedure. Explain the precision-recall trade-off.

Do not optimize a threshold using the test set and then report performance on that same test set as unbiased.

### 9. EXPLAINABILITY AND RESPONSIBLE ML

Provide useful model interpretation using suitable methods such as permutation feature importance and, where appropriate, interpretable model coefficients.

Requirements:

- Show which features influence model predictions.
- Explain the limitations of feature importance.
- Avoid causal claims based on correlations.
- Audit relevant subgroup performance when the data supports a responsible analysis.
- Document limitations arising from historical data, class imbalance, sampling, and potential bias.
- Avoid treating demographic or protected attributes as automatically valid lending criteria.
- Do not claim that a fairness audit proves the model is fair.
- Do not present predictions as definitive judgments about a person's financial worth.

The dashboard must display an educational-use disclaimer.

### 10. PREMIUM INTERACTIVE DASHBOARD

Build a polished, responsive Streamlit application with a consistent professional visual design.

The interface should feel like a modern financial analytics product rather than a default tutorial page.

Suggested product identity:

CREDIT INTELLIGENCE PRO
Subtitle: Explainable Credit Risk Analytics

Design direction:

- Clean dark-navy and light-neutral visual system.
- Restrained accent colours.
- Professional typography and spacing.
- Clear hierarchy and readable charts.
- Consistent cards, tables, labels, and navigation.
- Accessible colour contrast and useful error messages.
- Responsive behaviour where supported by Streamlit.

Create meaningful sections such as:

A. Executive Overview
- Dataset overview.
- Model status.
- Key validated evaluation metrics.
- Class distribution.
- Important project limitations.

B. Data Explorer
- Dataset preview.
- Feature summaries.
- Missing-value analysis.
- Interactive exploratory charts.

C. Model Performance
- Model comparison.
- ROC curves.
- Confusion matrix.
- Precision, recall, F1-score, and ROC-AUC.
- Threshold analysis supported by saved evaluation data.

D. Credit Risk Demo
- A validated input form using the selected model's expected features.
- Input validation and clear field descriptions.
- Estimated bad-credit probability, when the model supports a valid probability estimate.
- An educational risk category based on a documented, configurable threshold.
- A short explanation of influential features and prediction limitations.

E. Explainability
- Feature importance.
- Model limitations.
- Relevant subgroup evaluation results, when available.

F. About and Methodology
- Dataset source.
- Model methodology.
- Evaluation protocol.
- Disclaimer.
- Instructions for reproducibility.

Never use arbitrary hard-coded metrics to fill dashboard cards.

All displayed performance values must come from actual generated evaluation reports or saved artifacts.

The demo must never imply that an illustrative prediction is an actual bank decision or a real person's approved credit score.

### 11. SOFTWARE ENGINEERING AND AUTOMATED TESTING

Write meaningful automated tests using pytest.

Cover at least:

- Dataset loading and schema validation.
- Target encoding.
- Invalid and missing inputs.
- Preprocessing output.
- Train/test separation.
- Model training.
- Prediction output shape.
- Probability output and range validation, where supported.
- Metric calculations.
- Model serialization and loading.
- Inference using the saved pipeline.
- Dashboard input validation where practical.
- Error handling for missing or corrupted artifacts.

Add smoke tests for the main workflow.

Use deterministic test fixtures or small synthetic datasets for unit tests. Do not require a fresh network download for every test.

Run the actual test suite after implementation and fix genuine failures before proceeding.

Do not claim that tests passed unless the test command was executed and its result verified.

### 12. ERROR RECOVERY PROTOCOL

Follow this sequence whenever an error occurs:

1. Read the complete relevant error message.
2. Identify the root cause instead of guessing.
3. Inspect the relevant code, dependencies, paths, and input data.
4. Make the smallest safe correction.
5. Re-run the failing test or command.
6. Run related regression tests.
7. Record the issue and resolution in the implementation notes.

Do not hide errors with broad try/except blocks, disable failing tests, silently suppress warnings that indicate genuine problems, or change expected results merely to make tests pass.

Do not repeatedly reinstall packages when the problem is caused by an incorrect path, syntax error, missing file, or incompatible API.

If a tool freezes, a dependency is unavailable, or a task cannot be completed safely, preserve the workspace, report the exact blocker, and continue with independent tasks where possible.

### 13. REPRODUCIBILITY AND MODEL ARTIFACTS

The final workflow must support:

- Dataset validation.
- Reproducible training.
- Evaluation.
- Saving model artifacts.
- Loading the saved pipeline.
- Making predictions on valid new inputs.
- Launching the dashboard.

Persist only necessary artifacts.

Save evaluation results as machine-readable CSV or JSON files and charts as image files.

Use version-compatible model serialization. Treat serialized model files as trusted local artifacts only.

Never load untrusted pickle or joblib files.

Ensure the dashboard can distinguish missing artifacts from corrupted artifacts and provides actionable guidance.

### 14. PROFESSIONAL DOCUMENTATION AND GITHUB

Write a complete, professional README containing:

- Project overview.
- Problem statement.
- Objectives.
- Architecture.
- Dataset source and target mapping.
- Feature engineering and preprocessing.
- Model comparison methodology.
- Actual evaluation results.
- Explainability and responsible-use considerations.
- Project directory structure.
- Environment setup.
- Installation instructions.
- Training and testing commands.
- Dashboard launch command.
- Screenshots section with instructions for adding genuine screenshots.
- Limitations and future improvements.

Create a sensible .gitignore.

Never commit virtual environments, secrets, unnecessary caches, or large temporary files.

Include a clear dependency list and avoid adding packages that are not needed.

Prepare a GitHub-ready project, but do not push commits, publish a repository, or upload data without explicit user authorization.

Never fabricate certificates, internship completion, test results, screenshots, or claims about real-world deployment.

### 15. IMPLEMENTATION WORKFLOW

Work in controlled phases:

Phase 0 — Workspace inspection and recovery.

Phase 1 — Dataset validation and project configuration.

Phase 2 — Exploratory data analysis.

Phase 3 — Preprocessing and feature engineering.

Phase 4 — Baseline models and cross-validation.

Phase 5 — Model tuning and threshold analysis.

Phase 6 — Final model selection and one-time test evaluation.

Phase 7 — Explainability and responsible ML analysis.

Phase 8 — Streamlit dashboard and inference integration.

Phase 9 — Automated tests, integration checks, and bug fixing.

Phase 10 — README, reproducibility, and GitHub readiness.

At every phase:

- Inspect existing work first.
- Implement a coherent, working increment.
- Run the relevant tests and checks.
- Review generated files and actual outputs.
- Fix failures before moving forward.
- Report completed work, evidence, remaining tasks, and the next phase.
- Preserve a concise progress file so work can resume after a restart.

Do not generate every file blindly in one massive operation. Do not claim a phase is complete when only its code has been written.

Continue autonomously through phases when the environment is stable and the results are verified. If the application freezes, a critical decision requires user input, or a risky destructive operation is needed, stop safely at that boundary.

### 16. DEFINITION OF DONE

The project is complete only when:

- The documented dataset is validated.
- The complete training pipeline is reproducible.
- The required classifiers have been evaluated.
- The final evaluation follows the stated data-isolation rules.
- The model and preprocessing pipeline load successfully.
- The dashboard launches and performs a valid demo prediction.
- The actual generated reports and charts are available.
- Automated tests have been executed and their results documented.
- Critical defects have been fixed or clearly disclosed.
- The README contains accurate setup instructions and measured results.
- The repository is organized and suitable for a professional portfolio review.

If any criterion cannot be met, explicitly list it as incomplete rather than pretending the project is finished.

### 17. START NOW

Begin with Phase 0.

Inspect the actual Antigravity workspace and Python environment. Summarize the existing state, identify risks, and implement the safest next step.

Do not delete or overwrite existing work. Do not assume a previous phase succeeded without inspecting its artifacts.

Operate as a careful senior engineer: build, verify, document, and improve. Prioritize correctness, reproducibility, explainability, maintainability, and professional presentation over unnecessary complexity.

Your goal is a genuinely impressive, technically defensible Credit Scoring Model project—not merely a visually attractive demo.
