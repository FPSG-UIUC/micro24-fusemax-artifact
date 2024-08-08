PATH=/home/workspace/.local/bin:$PATH
pip install -r requirements.txt
cd src
python3 utils/check.py
