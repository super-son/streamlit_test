# streamlit run test_copy.py
import sklearn
import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import warnings
import pickle
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Custom Theme",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="auto",
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# with open('prac3_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

loaded_model = joblib.load('prac2_model.pkl')

with st.sidebar:
    st.image("ggilook.PNG")
col1, col2  = st.columns(2)

with col1:
    container1 = st.container(border=False)  
    container1.title(':violet[Personalize]')
    container1.title('your')         
    container1.title('journey:sunglasses:')
    container1.title('')

with col2:
    st.empty()

st.image("wangja.jpg", width=200)

container2 = st.container(border=True)

info1 = container2.text_input('계절', 1, help='봄:1, 여름:2, 가을:3, 겨울:4')
info2 = container2.text_input('성별', 1, help='남자:1, 여자:2')
info3 = container2.text_input('나이', 20 )
info4 = container2.text_input('여행스타일1',1, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info5 = container2.text_input('여행스타일2',2, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info6 = container2.text_input('여행스타일3',1, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info7 = container2.text_input('여행스타일4',2, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info8 = container2.text_input('여행스타일5',1, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info9 = container2.text_input('여행스타일6',2, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info10 = container2.text_input('여행스타일7',1, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
info11 = container2.text_input('여행스타일8',2, help='기쁘게:1, 슬프게:2,행복하게:3, 걱정없이:4')
traveler_list = [[info1,info2,info3,info4,info5,info6,info7,info8,info9,info10,info11]]
y_pred = loaded_model.predict(traveler_list)
y_pred_proba = loaded_model.predict_proba(traveler_list)
top_10_classes = np.argsort(-y_pred_proba, axis=1)[:, :10]
for i, sample_top_classes in enumerate(top_10_classes):
    top_10_predictions_list = []
    for class_index in sample_top_classes:
        top_10_predictions_list.append(loaded_model.classes_[class_index])


if st.button('Submit'):
    with st.spinner('여행지를 선택 중...'):
        time.sleep(2)
        st.write("저의 추천 관광지 10곳입니다:smile:")
        st.write(top_10_predictions_list)

# with st.sidebar:
#     messages = st.container(height=300)
#     if prompt := st.chat_input("Say something"):
#         messages.chat_message("user").write(prompt)
#         messages.chat_message("assistant").write(f"Echo: {prompt}")

# 지도그리기
# df = pd.DataFrame({
#     "col1": 35.1796,
#     "col2": 129.0756,
#     "col3": 50,
#     "col4": np.random.rand(1000, 4).tolist(),
# })
# st.map(df,
#     latitude='col1',
#     longitude='col2',
#     size='col3',
#     color='col4')
# st.info("""지도를 확인하세요""")
# st.title('')
