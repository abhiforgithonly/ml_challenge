import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = "dev-secret-key"
DEBUG = True

ALLOWED_HOSTS = ["*"]

ROOT_URLCONF = "train_and_app"

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
]

MIDDLEWARE = []

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {},
    }
]

WSGI_APPLICATION = None

DATABASES = {}

STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]
