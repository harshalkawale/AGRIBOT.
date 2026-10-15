# AgriBot — Smart Farming Assistant

A beginner-friendly Streamlit website for a college project.

## Features
- Marathi/English keyword-based farming chat
- Random Forest crop recommendation trained from a public crop dataset
- Crop-leaf photo upload and preview (not disease diagnosis yet)

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:
   `pip install -r requirements.txt`
4. Start the website:
   `streamlit run app.py`

## Publish online for free
1. Create a GitHub account and a new repository named `agribot`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app** and select your `agribot` repository.
5. Set the main file path to `app.py` and click **Deploy**.
6. After it starts, copy the `https://....streamlit.app` link and submit it.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Important limitations
- The crop model downloads a public CSV dataset on first use. Internet access is required.
- Crop predictions are educational suggestions, not guaranteed agronomic advice.
- Chat replies are rule-based, not a generative AI service.
- The disease tab only previews a photo. A trained image classification model must be added for genuine disease prediction.


## Verification note
The app source syntax was checked. The live dataset download and cloud deployment still need to be tested in an internet-connected environment.
