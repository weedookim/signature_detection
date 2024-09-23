
import streamlit as st
import tempfile
import fitz  # PyMuPDF
from fastai.vision.all import load_learner, PILImage
from PIL import Image

# PDF를 이미지로 변환하고 관심 영역(ROI)을 크롭하는 함수
def convert_pdf_to_image_and_crop(pdf_file, dpi=150):  # DPI 설정은 의미가 없으므로 기본값 유지
    # PDF 파일이 비어 있는지 확인
    if pdf_file.size == 0:
        st.error("Uploaded file is empty. Please upload a valid PDF file.")
        return None

    try:
        # PDF를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())  # Streamlit UploadedFile 객체를 파일로 저장
            tmp_file.flush()  # 데이터를 확실히 저장

        # PyMuPDF를 사용하여 PDF 파일 열기
        pdf_document = fitz.open(tmp_file.name)

        # 첫 번째 페이지 가져오기
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)  # DPI에 따라 이미지의 해상도 조정
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 관심 영역(ROI) 지정 (예: 우측 상단에서 가로 10cm, 세로 5cm 크롭)
        cm_to_inch = 2.54
        dpi_to_cm_ratio = dpi / cm_to_inch

        roi_width_px = int(15 * dpi_to_cm_ratio)  # 10cm를 픽셀로 변환
        roi_height_px = int(8 * dpi_to_cm_ratio)  # 5cm를 픽셀로 변환

        # 이미지 크기 구하기
        img_width, img_height = img.size

        # 관심 영역을 오른쪽 상단에서 크롭 (left, top, right, bottom)
        left = img_width - roi_width_px
        top = 0
        right = img_width
        bottom = roi_height_px

        # 크롭된 이미지
        roi_image = img.crop((left, top, right, bottom))

        return roi_image

    except Exception as e:
        st.error(f"Failed to process the PDF file: {e}")
        return None
