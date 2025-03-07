# import os

# print("Sanity check", flush=True)

# os.system('gunicorn --workers=4 --threads=4 --worker-class=gthread --bind 0.0.0.0:5000 -m 007 app:server')


from waitress import serve
from OCR.app import server  # Import your Dash app instance

print("Sanity check", flush=True)

serve(server, host="0.0.0.0", port=5000, threads=4)
