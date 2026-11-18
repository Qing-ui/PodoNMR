# PodoNMR

PodoNMR ranks eight diterpene scaffold classes and scores 28 structural motifs from assigned ¹³C and ¹H chemical shifts and carbon types. It retrieves similar compounds from a 315-compound reference index and visualizes motif co-occurrence.

## Run the application

Use Python 3.12. Clone the repository and run the following commands from the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Analyze a sample

Upload your assigned NMR measurements using **Upload sample table**, then click **Run analysis**. Use one UTF-8 CSV file per sample with the following columns.

| Column | Required value |
| --- | --- |
| `carbon_id` | Identifier for each assigned carbon site. Repeat the identifier for a second proton attached to the same carbon. |
| `delta_C` | Assigned ¹³C chemical shift in ppm. |
| `delta_H` | Assigned shift in ppm of a directly attached ¹H. Leave blank for carbons without an attached proton. |
| `C_type` | Accepted values are CH3, CH2, CH, Cq, C=O, COOR and CHO. |

Provide at least eight distinct carbon sites and one assigned proton shift. Rows for the same carbon must agree on its ¹³C shift and carbon type. You can label the results by adding a sample_id column with a single value.

## Interpret and export results

- **Overview** shows the scaffold ranking, top-scoring motifs, and related references.
- **Network** shows co-occurrence among the displayed motifs in known reference compounds. Network scores provide context. Ranking uses the fitted model scores.
- **Data** shows all eight scaffold scores, all 28 motif scores, and the nearest reference compounds. A smaller distance between standardized NMR descriptors indicates a closer match in the reference index.
- **Export** provides CSV tables, PNG and SVG network figures, a standalone HTML report, and a ZIP containing the full analysis.

Scaffold and motif scores are uncalibrated. Use them to prioritize candidates for structural review. The reference-domain indicator is a heuristic based on the top scaffold score and reference similarity. The overview shows up to ten motifs scoring at least 0.50. If every score is below 0.50, it displays the five highest scores.

## Deploy

On a Streamlit host, select Python 3.12 and set the entry point to app.py. Install dependencies from requirements.txt. The bundled fitted models, reference index, motif definitions, and network tables support inference and reference retrieval.
