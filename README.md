# PodoNMR

Analyze assigned carbon and proton NMR shifts to rank eight diterpene scaffold classes, recommend structural motifs from 28 targets, and retrieve similar reference compounds. The app also displays the sample-specific fragment co-occurrence network and exports the results.

## Run locally

Use Python 3.12. On Windows PowerShell, run these commands in the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open [PodoNMR](http://127.0.0.1:8501). Select **Example** and click **Run analysis** to reproduce the 3-oxo-abieta-13-ene case, or select **Upload CSV** and use the example file as the input template.

Required CSV columns are `carbon_id`, `delta_C`, `delta_H`, and `C_type`. Use one row for each assigned proton shift and repeat the carbon ID for two protons attached to the same carbon. Leave `delta_H` blank for carbons without attached protons. Accepted types are `CH3`, `CH2`, `CH`, `Cq`, `C=O`, `COOR`, and `CHO`.

The app uses the fitted model and its 315-compound reference index. Model scores are uncalibrated. Analysis downloads are available in **Export**.

## Deploy

Set Python to 3.12 and use `app.py` as the Streamlit entry point. Dependencies are listed in `requirements.txt`.
