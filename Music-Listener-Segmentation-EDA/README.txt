MUSIC LISTENER SEGMENTATION - DATA ANALYSIS & EDA  (Pavithra)

ONE FOLDER, EVERYTHING INSIDE

music_listeners.csv              the dataset
eda.py                           main script (cleaning + EDA + charts + clustering prep)
requirements.txt                 libraries
run.bat                          double-click to install and run (Windows)
Music_Listener_EDA_Report.docx   the Word documentation
eda_output/                      results (already generated, re-created every time you run eda.py)
    charts/                      9 charts
    music_listeners_cleaned.csv  give THIS to the modelling teammate (train_model.py)
    music_listeners_clustering_ready.csv   scaled copy, reference only
    cluster_preview_labels.csv   each listener with Casual / Explorer / Heavy preview
    findings.md, data_quality_report.txt, data_overview.txt

HOW TO RUN
1. VS Code: File > Open Folder > select this folder
2. Terminal > New Terminal, then:
       python -m pip install -r requirements.txt
       python eda.py
   (or just double-click run.bat)
