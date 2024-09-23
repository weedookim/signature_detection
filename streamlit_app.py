import streamlit as st
from pdf2image import convert_from_path
import tempfile
from fastai.vision.all import load_learner, PILImage
from PIL import Image

# PDF를 이미지로 변환하고 관심 영역(ROI)을 크롭하는 함수
def convert_pdf_to_image_and_crop(pdf_file, dpi=600):
    # PDF 파일이 비어 있는지 확인
    if pdf_file.size == 0:
        st.error("Uploaded file is empty. Please upload a valid PDF file.")
        return None

    try:
        # PDF를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())  # Streamlit UploadedFile 객체를 파일로 저장
            tmp_file.flush()  # 데이터를 확실히 저장

        # convert_from_path로 PDF 파일을 이미지로 변환
        images = convert_from_path(tmp_file.name, dpi=dpi)

        if not images:
            return None

        # 첫 번째 페이지의 이미지 가져오기
        image = images[0]

        # 관심 영역(ROI) 지정 (예: 우측 상단에서 가로 10cm, 세로 5cm 크롭)
        cm_to_inch = 2.54
        dpi_to_cm_ratio = dpi / cm_to_inch

        roi_width_px = int(15 * dpi_to_cm_ratio)  # 10cm를 픽셀로 변환
        roi_height_px = int(7 * dpi_to_cm_ratio)  # 5cm를 픽셀로 변환

        # 이미지 크기 구하기
        img_width, img_height = image.size

        # 관심 영역을 오른쪽 상단에서 크롭 (left, top, right, bottom)
        left = img_width - roi_width_px
        top = 0
        right = img_width
        bottom = roi_height_px

        # 크롭된 이미지
        roi_image = image.crop((left, top, right, bottom))

        return roi_image

    except Exception as e:
        #st.error(f"Failed to process the PDF: {e}")
        return None

# 서명 인식 모델을 사용해 예측하는 함수
def predict_signature_in_pdf(pdf_file, learner_path='processed_data/signature_detector.pkl', dpi=600):
    # PDF를 이미지로 변환하고 관심 영역(ROI)만 크롭
    cropped_image = convert_pdf_to_image_and_crop(pdf_file, dpi=dpi)
    
    if cropped_image is None:
        st.error("Failed to extract image from PDF.")
        return None, None
    
    # FastAI 학습된 모델 로드
    learner = load_learner(learner_path)
    
    # 이미지를 모델이 이해할 수 있는 형식으로 변환
    pil_img = PILImage.create(cropped_image)
    
    # 모델 예측
    prediction, pred_idx, probs = learner.predict(pil_img)
    
    return prediction, probs[pred_idx].item()

# Streamlit 앱 시작
st.title("Signature Detection in PDF")
st.write("Upload a PDF file to detect the presence of a signature.")

# PDF 파일 업로드 인터페이스
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# PDF 파일이 업로드된 경우 처리
if uploaded_file is not None:
    st.write("PDF file uploaded successfully!")
    
    # 서명 인식 예측
    prediction, confidence = predict_signature_in_pdf(uploaded_file)

    # 예측 결과 출력
    if prediction is not None:
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence:.2f}")
    
    # 관심 영역 이미지를 표시
    cropped_image = convert_pdf_to_image_and_crop(uploaded_file)
    if cropped_image:
        st.image(cropped_image, caption="Cropped ROI from PDF", use_column_width=True)
else:
    st.write("Please upload a PDF file.")
