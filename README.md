# Early Prediction of Student Outcomes in Online Learning (OULAD)

Summative assignment: Model Training and Evaluation (Machine Learning, ALU).

This project predicts whether an online-learning student will finish a module with **Distinction, Pass, Fail or Withdrawn**, using only the information available in the **first 60 days** of the module. The aim is an early-warning signal: a prediction is only useful for support if it arrives while there is still time to act.

## Links

| Item | Link |
|---|---|
| Written report | `<ADD LINK>` |
| Google Drive folder (raw dataset, saved models, logs) | `<ADD LINK>` |
| Video presentation | `<ADD LINK>` |

## Dataset

**Open University Learning Analytics Dataset (OULAD)**: 32,593 student-module records, 7 modules, and about 10.6 million rows of daily clicks on the virtual learning environment, together with demographics and assessment results.

- Source: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset (also published by the Open University)
- Licence: CC BY 4.0
- Citation: Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

The raw data is **not** stored in this repository. It is in the Google Drive folder above, and the notebook downloads it automatically (or asks you to upload the zip file).

## Approach

1. **Leakage-safe problem design.** Only behaviour up to day 60 is used. Students who had already unregistered before day 60 are removed, because their outcome is already known. After this filter 26,361 students remain.
2. **Feature engineering** (per student per module): demographics and course context, click-stream engagement (totals, active days, recency, trend, activity types), and assessment behaviour (submissions, scores, lateness).
3. **Grouped split by student** (about 70 / 15 / 15), so no student appears in more than one split. The preprocessing pipeline is fitted on the training set only.
4. **Models.** Traditional ML (scikit-learn): Logistic Regression, Random Forest, Gradient Boosting, SVM. Deep learning (TensorFlow/Keras): Sequential and Functional MLPs fed by a `tf.data` pipeline.
5. **13 experiments**, all scored on the validation set, with hyperparameter search using grouped cross-validation, class weights for the imbalance, and a feature ablation. Every run is logged with Weights & Biases in offline mode.
6. **Final evaluation:** the two best models (chosen on validation macro-F1) are evaluated once on the test set, with confusion matrices, ROC and precision-recall curves, learning curves, error analysis and a multi-seed repeat to estimate run-to-run noise.

## Results (test set)

| Model | Macro-F1 | Accuracy | Macro-AUC |
|---|---|---|---|
| Random Forest (tuned) | 0.489 | 0.534 | 0.782 |
| MLP Sequential (class weights) | 0.481 | 0.502 | 0.787 |

Predicting "Pass" for everyone would give a macro-F1 of about 0.16. Using only demographics and course context (no behaviour features), a Random Forest reaches a validation macro-F1 of only 0.29, so the engineered behaviour features carry most of the signal. The full experiment table is in `experiment_results.csv` (in the Drive folder) and in the notebook (section 11).

*(Update these numbers if you re-run the notebook.)*

## Repository contents

| File | Description |
|---|---|
| `oulad_early_prediction.ipynb` | The full, documented notebook (data, features, models, experiments, evaluation) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## How to run

1. Open `oulad_early_prediction.ipynb` in Google Colab (or Jupyter with a recent Python 3).
2. Choose **Runtime, then Run all**. When asked, upload the OULAD zip file (downloaded from the UCI link above) if the automatic download does not work.
3. The full run takes roughly 25 to 30 minutes on Colab's free CPU. The last cell saves the best models and copies them to Google Drive (it asks for permission).

Locally: `pip install -r requirements.txt`, then run the notebook from top to bottom. The last cell uses Google Colab's Drive mounting and can be skipped outside Colab.

**Reproducibility.** Random seeds are fixed in the notebook (`SEED = 42`). Exact numbers can still vary slightly between machines and library versions, especially for the neural networks. The multi-seed cell in section 12 quantifies this variation.

## Limitations

- The data comes from a single UK distance-learning university, so results may not transfer to other settings.
- Only 60 days of information are used by design, which limits how well later outcomes can be predicted.
- Predictions use demographic variables, which raises fairness questions that are not fully analysed here.
- The Withdrawn class is the hardest to predict, and the neural-network results are sensitive to random seeds.

## Acknowledgements

**Dataset.** The data comes from the Open University Learning Analytics Dataset (OULAD), released by the Open University under CC BY 4.0 (Kuzilek, Hlosta and Zdrahal, 2017). Thank you to the authors and the Open University for making it public.

**AI assistance.**It helped me to:
- debug the notebook code 
- suggest the set of experiments;
- explain parts of the code and the results to me.

I ran the notebook myself, checked the outputs, and wrote the report, the notes in the notebook and the video presentation in my own words. All results reported here come from my own runs.

**Libraries and tools.** scikit-learn, TensorFlow/Keras, pandas, NumPy, Matplotlib, Seaborn, SciPy, joblib and Weights & Biases (offline mode), run on Google Colab.

**Collaboration.**
