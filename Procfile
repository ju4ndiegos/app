web: python manage.py collectstatic --noinput && gunicorn apk_detector.wsgi --bind 0.0.0.0:$PORT --workers 1 --timeout 120
