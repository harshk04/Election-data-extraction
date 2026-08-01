# Electoral Roll OCR

Production-grade scaffold for an Electoral Roll OCR pipeline that processes scanned electoral roll PDFs and extracts voter records into structured JSON.

## Stack

- Python 3.12
- PyMuPDF
- OpenCV
- PaddleOCR
- Pydantic
- Pandas

## Project Structure

```text
app/
    config/
    services/
    models/
    utils/

data/
    pdfs/
    pages/
    crops/

outputs/

tests/
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

PaddleOCR runtime note:

- Use Python `3.12`. This repo is not configured for Python `3.13`.
- Install the PaddlePaddle runtime package `paddlepaddle`; do not install the unrelated package named `paddle`.
- If your environment already has `paddle`, remove it first with `pip uninstall paddle`.
- Then install a compatible `paddlepaddle` build for your platform by following the official PaddlePaddle installation guide: https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html
- The app defaults to Hindi OCR with models cached under `outputs/paddleocr` inside the project.

## Environment Configuration

The application reads configuration from `.env` using `pydantic-settings`.

Example variables:

```env
APP_NAME=Electoral Roll OCR
APP_ENV=development
APP_DEBUG=true
LOG_LEVEL=INFO
DATA_DIR=data
PDFS_DIR=data/pdfs
PAGES_DIR=data/pages
CROPS_DIR=data/crops
OUTPUTS_DIR=outputs
```

## Current Scope

## Running

Basic run:

```bash
python main.py
```

## What It Does

- Renders PDF pages to PNG
- Preprocesses page images
- Detects entry grid boxes
- Crops voter entries
- Detects deleted entries
- Runs PaddleOCR
- Extracts structured voter fields
- Validates extracted records
- Writes JSON output
