 One-Click Pack
---------------------------
Follow these steps (Windows or Linux/Mac):

1) Create virtual env and activate
   Windows: python -m venv venv & venv\Scripts\activate
   Linux/Mac: python3 -m venv venv && source venv/bin/activate

2) Install requirements:
   pip install -r requirements.txt

3) Prepare data as explained in the canvas doc (train/val folders).

4) Run soil model training:
   python soil_model_train.py
5) Run image model training (this will save crop_model.h5):
   python train_model.py
6) Start demo UI:
   streamlit run streamlit_app.py

If you get errors, paste them in chat and I'll help.
