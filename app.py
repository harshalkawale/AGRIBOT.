import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from io import BytesIO
import requests

st.set_page_config(page_title="AgriBot | Smart Farming", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #f4fbf3 0%, #ffffff 55%);}
h1, h2, h3 {color: #175c35;}
div[data-testid="stMetric"] {background:#edf8ee; padding:14px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/harshavardhanrallapalli/crop-recommendation/master/Crop_recommendation.csv"
FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

@st.cache_resource(show_spinner="Loading agricultural data and training the crop model...")
def train_crop_model():
    response = requests.get(DATA_URL, timeout=20)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content))
    df.columns = [str(c).strip().lower() for c in df.columns]
    expected = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall", "label"]
    if not set(expected).issubset(df.columns):
        raise ValueError("The dataset columns do not match the expected crop recommendation format.")
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, sorted(y.unique().tolist())

def chatbot_reply(message):
    q = message.lower().strip()
    answers = [
        (["hello", "hi", "namaskar", "नमस्कार", "नमस्ते"], "नमस्कार! 🌱 मी AgriBot आहे. तुम्ही पीक निवड, माती, सिंचन किंवा पिकांच्या रोगांविषयी विचारू शकता."),
        (["crop", "पीक", "पिकाची निवड", "which crop"], "पीक सुचवण्यासाठी Crop Prediction विभागात N, P, K, तापमान, आर्द्रता, मातीचा pH आणि पाऊस यांची माहिती भरा."),
        (["fertilizer", "खत", "fertiliser"], "खत वापरण्यापूर्वी माती परीक्षण करून घ्या. योग्य खत आणि प्रमाण हे पीक, वाढीची अवस्था आणि मातीच्या अहवालावर अवलंबून असते."),
        (["water", "irrigation", "पाणी", "सिंचन"], "सिंचनाचे प्रमाण पीक, मातीचा प्रकार, हवामान आणि वाढीच्या अवस्थेनुसार ठरते. शेतात पाणी साचू देऊ नका आणि स्थानिक कृषी सल्ला घ्या."),
        (["disease", "रोग", "leaf", "पान"], "Disease Assistant विभागात पानाचा फोटो अपलोड करू शकता. या starter version मध्ये फोटो साठवून/पाहून घेता येतो; AI रोग निदान अद्याप जोडलेले नाही. निदानासाठी कृषी तज्ज्ञांचा सल्ला घ्या."),
        (["soil", "माती", "ph"], "मातीचा pH, N-P-K आणि माती परीक्षणाचा अहवाल पीक निवड व खत व्यवस्थापनासाठी उपयोगी असतो."),
        (["rain", "rainfall", "पाऊस", "हवामान"], "पाऊस आणि तापमान पीक निवडीवर परिणाम करतात. पेरणीपूर्वी स्थानिक हवामान अंदाज आणि कृषी विभागाचा सल्ला तपासा.")
    ]
    for keywords, answer in answers:
        if any(word in q for word in keywords):
            return answer
    return ("मी सध्या मूलभूत शेतीविषयक प्रश्नांना नियमाधारित उत्तरे देतो. "
            "पीक सुचवण्यासाठी Crop Prediction विभाग वापरा. विशिष्ट सल्ल्यासाठी तुमचे पीक, "
            "जिल्हा आणि समस्या यांची माहिती द्या.")

st.title("🌱 AgriBot")
st.caption("Your Smart Farming Assistant | Crop Recommendation • Farming Chat • Leaf Photo Demo")
st.info("Educational college project. Crop recommendations are model outputs, not guaranteed agricultural advice.")

tab_chat, tab_crop, tab_disease, tab_about = st.tabs(
    ["💬 Farming Chat", "🌾 Crop Prediction", "🍃 Disease Assistant", "ℹ️ About"]
)

with tab_chat:
    st.subheader("Ask AgriBot")
    st.write("शेतीविषयक प्रश्न विचारा — Marathi किंवा English.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "नमस्कार! मी AgriBot आहे. तुम्हाला कोणती मदत हवी आहे?"}
        ]
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
    prompt = st.chat_input("उदा. Which fertilizer should I use? / खताबद्दल माहिती")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        answer = chatbot_reply(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab_crop:
    st.subheader("Crop Recommendation")
    st.write("Enter values from your soil test and local weather data.")
    try:
        model, model_accuracy, crop_names = train_crop_model()
        st.metric("Test-set accuracy", f"{model_accuracy * 100:.1f}%")
        st.caption("Accuracy is measured on a held-out split of the public dataset; it does not guarantee results in your field.")
        with st.form("crop_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
                p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
                k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0)
            with c2:
                temperature = st.number_input("Temperature (°C)", min_value=-5.0, max_value=60.0, value=25.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
            with c3:
                ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0)
            submitted = st.form_submit_button("🌱 Predict suitable crop", type="primary")
        if submitted:
            input_df = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=FEATURES)
            prediction = model.predict(input_df)[0]
            st.success(f"Recommended crop from the model: **{str(prediction).title()}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                ranked = sorted(zip(model.classes_, probs), key=lambda item: item[1], reverse=True)[:3]
                st.write("Top model suggestions:")
                st.table(pd.DataFrame([{"Crop": str(label).title(), "Model score": f"{prob*100:.1f}%"} for label, prob in ranked]))
            st.warning("Use a soil-test report, local season, irrigation availability, market demand and advice from your agriculture officer before planting.")
    except Exception as e:
        st.error("The crop dataset could not be loaded, so the model could not be trained. Check your internet connection and try again.")
        st.caption(f"Technical detail: {e}")

with tab_disease:
    st.subheader("Crop Leaf Photo Assistant")
    st.write("Upload a clear leaf photo. This starter version previews the image; it does not diagnose disease.")
    uploaded = st.file_uploader("Choose a leaf image", type=["png", "jpg", "jpeg"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
            st.warning("No disease prediction model is connected yet. Image appearance alone is not enough for a reliable diagnosis.")
            st.markdown("**For a real disease-detection feature:** train or integrate a validated plant-leaf image classifier, test it on local crop images, and show its confidence and limitations.")
        except Exception:
            st.error("Could not read that image. Please upload a JPG or PNG file.")

with tab_about:
    st.subheader("About this project")
    st.write("""
    **AgriBot** is a student project that demonstrates:
    - A rule-based farming chat assistant
    - A Random Forest crop recommendation model trained on a public crop dataset
    - A crop-leaf image upload interface for a future disease classifier
    """)
    st.write("**Technologies:** Python, Streamlit, Pandas, Scikit-learn, Pillow.")
    st.write("**Limitations:** The chat assistant is not a generative AI model, and leaf disease detection is not implemented in this starter version.")
    st.write("Always confirm farming decisions with local agricultural experts and soil-test information.")
