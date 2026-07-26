FROM python:3.11-slim

WORKDIR /app

# Install dependencies FIRST, before copying source, so that changing
# application code doesn't invalidate this layer and force a full
# dependency reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir flask python-dotenv pillow

# Now copy the source. (torch/torchvision are deliberately NOT installed
# in the image - they're only needed to run ml/train_classifier.py once,
# locally, to produce certificate_model.pt. Keeping them out of the image
# keeps it small; the web app runs fine without them and just reports
# the certificate-check feature as unavailable if the model file is
# missing.)
COPY . .

# No secrets or environment-specific values are baked into the image -
# they are supplied at run time via environment variables / .env
# (see .env.example). Anything placed inside an image layer stays
# visible in the image history even if later "removed".
ENV FLASK_APP=app.py
EXPOSE 5000

CMD ["python", "app.py"]
