#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1WqZu3SULybgCaLwBaXIkO_x04GkmI5Us'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
#st.cache_data

def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지",use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(2)

    # 1st Row - Images
    for i in range(2):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}",use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(2):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(2):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("관련 정보를 찾고 있어요... 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("관련 정보를 찾았어요!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://img.cjnews.cj.net/wp-content/uploads/2021/08/tvN%EA%B0%AF%EB%A7%88%EC%9D%84%EC%B0%A8%EC%B0%A8%EC%B0%A8_12_%EB%A9%94%EC%9D%B8%ED%8F%AC%EC%8A%A4%ED%84%B0%EA%B3%B5%EA%B0%9C-725x1024.jpg",
            "https://th.bing.com/th/id/OIP.Mf7R5QdS3zqfXUGIaFAp9wHaEK?rs=1&pid=ImgDetMain",
        ],
        'videos': [
            "https://www.youtube.com/watch?v=wMuIIU66udM&list=PLaw_8D5aLtDiwKE011fRa6HSK98LZupo_",
            "https://www.youtube.com/watch?v=nRjg0Hcb1b4&list=PLaw_8D5aLtDiwKE011fRa6HSK98LZupo_&index=2",
        ],
        'texts': [
            "소나기 없는 인생이 어딨겠어.-두식",
            "갯마을 차차차 관련 영상 입니다.",
        ]
    },
    labels[1]: {
        'images': [
            "https://th.bing.com/th/id/R.660eed9ebedff0818132f6423e764527?rik=bCS5XtVCkzuTaw&riu=http%3a%2f%2fcdn.ggilbo.com%2fnews%2fphoto%2f202001%2f740746_575617_5420.jpg&ehk=OwZpgmLGhWhTiSratUe8ELLG8UKZxY8BpVD7LYtRjnY%3d&risl=&pid=ImgRaw&r=0",
            "https://th.bing.com/th/id/OIP.BAkNjT1qb41o4dC54UvTbgHaJQ?rs=1&pid=ImgDetMain2",

        ],
        'videos': [
            "https://www.youtube.com/watch?v=Q5DITF2ZXLw",
            "https://www.youtube.com/watch?v=9jnsUMjuP84",

        ],
        'texts': [
            "오래오래 따뜻하고 싶어요.-동백",
            "동백꽃필무렵 관련 영상 입니다.",

        ]
    },
    labels[2]: {
        'images': [
            "https://i.pinimg.com/originals/5f/8d/2d/5f8d2d8a5941b5726363311b0d954559.jpg",
            "https://image.mediapen.com/news/201810/news_392574_1540955380_m.jpg",

        ],
        'videos': [
            "https://www.youtube.com/watch?v=bs7JXCmz3j8",
            "https://www.youtube.com/watch?v=xVZ_QTc8azU",

        ],
        'texts': [
            "너의 낭군으로 살았던 그 백일 간은 내게 모든 순간이 기적이었다.-이율",
            "백일의 낭군님 관련 영상 입니다..",

        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

