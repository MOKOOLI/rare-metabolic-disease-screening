install:
	pip install -r requirements.txt
run:
	python run_pipeline.py
quick:
	python run_pipeline.py --quick
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
