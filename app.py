import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance
import io

st.set_page_config(page_title="Tách Nền & Tối Ưu Chữ Ký", layout="wide")
st.title("🖼️ Ứng dụng Bắp Tách Nền & Tối Ưu Ảnh")

# Hàm tối ưu hóa chữ ký
def enhance_signature(transparent_image, contrast_factor, sharpness_factor):
    # Đảm bảo ảnh có kênh alpha (trong suốt)
    if transparent_image.mode != 'RGBA':
        transparent_image = transparent_image.convert('RGBA')
    
    r, g, b, a = transparent_image.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    
    enhancer_contrast = ImageEnhance.Contrast(rgb_img)
    enhanced_rgb = enhancer_contrast.enhance(contrast_factor)
    
    enhancer_sharpness = ImageEnhance.Sharpness(enhanced_rgb)
    enhanced_rgb = enhancer_sharpness.enhance(sharpness_factor)
    
    r_e, g_e, b_e = enhanced_rgb.split()
    return Image.merge('RGBA', (r_e, g_e, b_e, a))

# Tải file lên
uploaded_file = st.file_uploader("Chọn ảnh chữ ký hoặc ảnh cần tách nền...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Ảnh gốc của bác sĩ", use_container_width=True)
        
    with col2:
        with st.spinner("Đang tách nền bằng AI (siêu tốc)..."):
            try:
                # Tách nền bằng rembg (nhẹ và không lo hết RAM)
                output_image = remove(input_image)
                
                st.markdown("### 🛠️ Bộ lọc tối ưu chữ ký")
                contrast = st.slider("Độ tương phản (làm rõ nét mực)", 1.0, 5.0, 2.5, 0.5)
                sharpness = st.slider("Độ sắc nét", 1.0, 4.0, 2.0, 0.5)
                scale = st.slider("Kích thước ảnh đầu ra (%) - Giảm để nhẹ file", 10, 100, 100, 10)
                
                # Áp dụng bộ lọc
                final_signature = enhance_signature(output_image, contrast, sharpness)
                
                # Thu nhỏ nếu cần giảm dung lượng
                if scale < 100:
                    new_width = int(final_signature.width * scale / 100)
                    new_height = int(final_signature.height * scale / 100)
                    final_signature = final_signature.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                st.image(final_signature, caption="Chữ ký đã tách nền & tối ưu sắc nét", use_container_width=True)
                
                # Nút tải file
                buf = io.BytesIO()
                final_signature.save(buf, format="PNG")
                st.download_button(
                    label="📥 Tải chữ ký sắc nét về máy (PNG)",
                    data=buf.getvalue(),
                    file_name="chuky_optimized.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
