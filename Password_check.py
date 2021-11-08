import os

EMAIL_USER = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASS = os.environ.get('EMAIL_PASSWORD')

print(EMAIL_PASS)