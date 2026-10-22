import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from transformers import AutoModelForImageSegmentation
import io
import gc

st.set_page_config(page_title="BiRefNet Signature & Background Remover", layout="wide")
st.title("🖼️ Ứng dụng Bắp Tách Nền & Tối Ưu Ảnh")

# Khởi tạo mô hình (cache để không tải lại nhiều lần)
@st.cache_resource
def load_model():
    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
    st.success("Đã tải mô hình AI tách nền thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")

# Hàm xử lý tách nền
def remove_background(image, model, device):
    orig_w, orig_h = image.size
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)), # Đã giảm xuống 256 để chống tràn RAM
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform_image(image.convert("RGB")).unsqueeze(0).to(device)

    gc.collect()

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(pred).resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    output_image = image.convert("RGB").copy()
    output_image.putalpha(mask_pil)
    
    del input_tensor, preds, pred
    gc.collect()
    
    return output_image, mask_pil

# Hàm tối ưu hóa chữ ký
def enhance_signature(transparent_image, contrast_factor, sharpness_factor):
    if transparent_image.mode != 'RGBA':
        return transparent_image
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
        with st.spinner("Đang tách nền bằng AI..."):
            try:
                output_image, mask_pil = remove_background(input_image, model, device)
                
                st.markdown("### 🛠️ Bộ lọc tối ưu chữ ký")
                contrast = st.slider("Độ tương phản (làm rõ nét mực)", 1.0, 5.0, 2.5, 0.5)
                sharpness = st.slider("Độ sắc nét", 1.0, 4.0, 2.0, 0.5)
                scale = st.slider("Kích thước ảnh đầu ra (%) - Giảm để nhẹ file", 10, 100, 100, 10)
                
                final_signature = enhance_signature(output_image, contrast, sharpness)
                
                # Thu nhỏ ảnh nếu người dùng giảm thanh trượt kích thước
                if scale < 100:
                    new_width = int(final_signature.width * scale / 100)
                    new_height = int(final_signature.height * scale / 100)
                    final_signature = final_signature.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                st.image(final_signature, caption="Chữ ký đã tách nền & tối ưu sắc nét", use_container_width=True)
                
                buf = io.BytesIO()
                final_signature.save(buf, format="PNG")
                st.download_button(
                    label="📥 Tải chữ ký sắc nét về máy (PNG)",
                    data=buf.getvalue(),
                    file_name="chuky_optimized.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error("Máy chủ miễn phí bị quá tải RAM. Vui lòng thử lại với một bức ảnh có dung lượng nhỏ hơn.")
