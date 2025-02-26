FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY wait-for-it.sh /app/wait-for-it.sh

COPY . /app/

RUN chmod +x /app/wait-for-it.sh
RUN chmod +x /app/manage.py

ENV PYTHONUNBUFFERED 1

CMD ["/app/wait-for-it.sh", "db:3306", "--", "sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]