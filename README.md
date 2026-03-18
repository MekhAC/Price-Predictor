# Used Car Price Predictor UI

This project now includes a Streamlit UI for:

- Single car prediction (base price + demand adjusted price)
- Batch prediction from CSV/Excel upload using the same batch pipeline

## Run UI

From the project root:

```powershell
python.exe -m streamlit run src/ui_app.py
```

If needed, install dependencies first:

```powershell
python.exe -m pip install -r requirements.txt
```

## What the UI does

1. Single Prediction tab
- Enter details for one car
- Shows:
	- Base Price (model output)
	- Demand Adjusted Price (after multipliers)
	- Composite demand multiplier
	- Demand breakdown table

2. Batch Prediction tab
- Upload `.csv`, `.xlsx`, or `.xls`
- Runs `src/batch_predict.py` logic internally
- Displays output table with all generated columns
- Lets you download the full predicted file
- Shows batch logs (metrics/summary) in an expandable section
