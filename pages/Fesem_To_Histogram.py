import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, filters, morphology, exposure
from skimage.color import rgb2gray

# Conversion factor (example: 1 pixel = 10 nm, adjust as needed)
PIXEL_TO_NM = 10

# Function to process and analyze FESEM image
def process_fesem_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Enhance contrast using adaptive histogram equalization
    image = exposure.equalize_adapthist(image) * 255
    image = image.astype(np.uint8)
    
    # Generate grayscale histogram with normalized values
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    
    # Apply Otsu’s thresholding for accurate segmentation
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    binary_image = morphology.closing(binary_image, morphology.disk(3))
    binary_image = (binary_image * 255).astype(np.uint8)
    
    # Label connected components
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image)
    
    # Extract fiber radius, orientation, and pore area
    sizes = [r.equivalent_diameter * PIXEL_TO_NM for r in regions]
    angles = [r.orientation * 180 / np.pi for r in regions]
    pore_areas = [r.area * (PIXEL_TO_NM ** 2) for r in regions]
    
    stats = {
        "Mean Size (nm)": np.mean(sizes) if sizes else 0,
        "Median Size (nm)": np.median(sizes) if sizes else 0,
        "Min Size (nm)": np.min(sizes) if sizes else 0,
        "Max Size (nm)": np.max(sizes) if sizes else 0,
        "Std Dev (nm)": np.std(sizes) if sizes else 0
    }
    
    return image, hist, bins, binary_image, sizes, angles, pore_areas, stats, labeled_image

st.set_page_config(page_title="FESEM Image Analyzer", layout="wide")
st.title("🔬 FESEM Image Analysis")
st.markdown("### Convert FESEM images to histograms, analyze fiber radius, orientation, and pore areas.")

uploaded_file = st.file_uploader("📤 Upload a FESEM Image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image, hist, bins, binary_image, sizes, angles, pore_areas, stats, labeled_image = process_fesem_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="🖼 Enhanced Original Image", use_container_width=True, channels="GRAY")
    with col2:
        st.image(binary_image, caption="📊 Segmented Image (Otsu’s Thresholding)", use_container_width=True, clamp=True)
    
    # Segmentation overlay
    outlines = measure.find_contours(binary_image, 0.5)
    overlay_image = np.copy(image)
    for outline in outlines:
        for x, y in outline:
            overlay_image[int(x), int(y)] = 255
    st.image(overlay_image, caption="🔍 Segmentation Outlines Overlay", use_container_width=True, channels="GRAY")
    
    # Grayscale histogram
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=bins[:-1], y=hist, mode='lines', fill='tozeroy', line=dict(color='black')))
    fig1.update_layout(title="Grayscale Intensity Histogram", xaxis_title="Pixel Intensity", yaxis_title="Normalized Frequency")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Fiber radius histogram
    fig2 = px.histogram(sizes, nbins=30, labels={'value': "Fiber Radius (nm)"}, title="Fiber Radius Distribution", color_discrete_sequence=["blue"], marginal="box")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Fiber angle histogram
    fig3 = px.histogram(angles, nbins=30, labels={'value': "Fiber Angle (°)"}, title="Fiber Angle Distribution", color_discrete_sequence=["red"], marginal="box")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Pore area histogram
    fig4 = px.histogram(pore_areas, nbins=30, labels={'value': "Pore Area (µm²)"}, title="Pore Area Distribution", color_discrete_sequence=["green"], marginal="box")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("### 📊 Statistical Data")
    st.write(stats)
    
    st.success("✅ Analysis Completed with Enhanced Visualizations!")








# import streamlit as st
# import cv2
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from skimage import measure, filters, morphology, exposure
# from skimage.color import rgb2gray

# # Conversion factor (example: 1 pixel = 10 nm, adjust as needed)
# PIXEL_TO_NM = 10

# # Function to process and analyze FESEM image
# def process_fesem_image(uploaded_file):
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
#     # Enhance contrast using adaptive histogram equalization
#     image = exposure.equalize_adapthist(image) * 255
#     image = image.astype(np.uint8)
    
#     # Generate grayscale histogram with normalized values
#     hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    
#     # Apply Otsu’s thresholding for accurate segmentation
#     threshold = filters.threshold_otsu(image)
#     st.write(f"Calculated Otsu Threshold: {threshold}")  # Debugging
#     binary_image = image > threshold
#     binary_image = morphology.closing(binary_image, morphology.disk(3))
#     binary_image = (binary_image * 255).astype(np.uint8)  # Ensure proper scaling
    
#     # Label connected components
#     labeled_image = measure.label(binary_image)
#     regions = measure.regionprops(labeled_image)
    
#     # Extract particle/nanofiber size distribution in nm
#     sizes = [r.equivalent_diameter * PIXEL_TO_NM for r in regions]
    
#     # Compute statistical data
#     size_mean = np.mean(sizes) if sizes else 0
#     size_median = np.median(sizes) if sizes else 0
#     size_min = np.min(sizes) if sizes else 0
#     size_max = np.max(sizes) if sizes else 0
#     size_std = np.std(sizes) if sizes else 0
    
#     stats = {
#         "Mean Size (nm)": size_mean,
#         "Median Size (nm)": size_median,
#         "Min Size (nm)": size_min,
#         "Max Size (nm)": size_max,
#         "Std Dev (nm)": size_std
#     }
    
#     return image, hist, bins, binary_image, sizes, stats, labeled_image

# # Streamlit UI Enhancements
# st.set_page_config(page_title="FESEM Image Analyzer", layout="wide")
# st.title("🔬 FESEM Image to Histogram Converter")
# st.markdown("### Analyze FESEM images with advanced histograms, statistical insights, and precise segmentation techniques.")

# uploaded_file = st.file_uploader("📤 Upload a FESEM Image", type=["png", "jpg", "jpeg", "tif"])

# if uploaded_file is not None:
#     image, hist, bins, binary_image, sizes, stats, labeled_image = process_fesem_image(uploaded_file)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="🖼 Enhanced Original Image", use_column_width=True, channels="GRAY")
#     with col2:
#         st.image(binary_image, caption="📊 Segmented Image (Otsu’s Thresholding)", use_column_width=True, clamp=True)
    
#     # Overlay segmentation outlines
#     outlines = measure.find_contours(binary_image, 0.5)
#     overlay_image = np.copy(image)
#     for outline in outlines:
#         for x, y in outline:
#             overlay_image[int(x), int(y)] = 255  # White outline
#     st.image(overlay_image, caption="🔍 Segmentation Outlines Overlay", use_column_width=True, channels="GRAY")
    
#     # Plot grayscale histogram with Plotly
#     fig1 = go.Figure()
#     fig1.add_trace(go.Scatter(x=bins[:-1], y=hist, mode='lines', fill='tozeroy', line=dict(color='black')))
#     fig1.update_layout(title="Grayscale Intensity Histogram", xaxis_title="Pixel Intensity", yaxis_title="Normalized Frequency")
#     st.plotly_chart(fig1, use_container_width=True)
    
#     # Plot size distribution histogram with Plotly (converted to nm)
#     if sizes:
#         fig2 = px.histogram(sizes, nbins=30, labels={'value': "Size (nm)"}, title="Particle/Nanofiber Size Distribution", 
#                             color_discrete_sequence=["blue"], marginal="box")
#         st.plotly_chart(fig2, use_container_width=True)
        
#         # Advanced Violin Plot for better size distribution visualization
#         fig3 = px.violin(y=sizes, box=True, points="all", title="Nanofiber Size Distribution Violin Plot", labels={'y': 'Size (nm)'} )
#         st.plotly_chart(fig3, use_container_width=True)
        
#         # Additional Box Plot for deeper insights into size distribution
#         fig4 = px.box(y=sizes, title="Box Plot of Nanofiber Sizes", labels={'y': 'Size (nm)'}, points="all")
#         st.plotly_chart(fig4, use_container_width=True)
        
#         # Plot Mean and Median Size as Horizontal Line Chart
#         fig5 = go.Figure()
#         fig5.add_trace(go.Scatter(x=[0, 1], y=[stats["Mean Size (nm)"]] * 2, mode='lines', 
#                                   line=dict(color='red', width=3, dash='dash'), name='Mean Size'))
#         fig5.add_trace(go.Scatter(x=[0, 1], y=[stats["Median Size (nm)"]] * 2, mode='lines', 
#                                   line=dict(color='blue', width=3, dash='dot'), name='Median Size'))
#         fig5.update_layout(title="Mean and Median Nanofiber Size Representation", xaxis_title="", yaxis_title="Size (nm)", showlegend=True)
#         st.plotly_chart(fig5, use_container_width=True)
        
#         # Display statistical insights
#         st.markdown("### 📊 Statistical Data")
#         st.write(stats)
    
#     st.success("✅ Analysis Completed with Optimized Accuracy and Enhanced Visualizations!")
