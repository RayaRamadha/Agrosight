# Pakai environment Python 3.10
FROM python:3.10

# Bikin folder kerja di dalam server
WORKDIR /code

# Copy requirements dan install library-nya
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy semua file (HTML, model, app.py) ke dalam server
COPY . .

# Jalankan aplikasi pakai Gunicorn (karena kamu masukin di requirements)
# Wajib pakai port 7860 untuk Hugging Face
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
