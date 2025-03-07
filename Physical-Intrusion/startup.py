import os

os.system('gunicorn --workers=4 --threads=4 --worker-class=gthread --bind 0.0.0.0:8050 -m 007 app:server &')
os.system('python3 PhysicalIntrusion.py')

