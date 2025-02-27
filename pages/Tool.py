import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from skimage import measure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Function to convert image into different types
def convert_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
    multispectral = cv2.merge([grayscale, binary, np.zeros_like(binary)])
    return binary, grayscale, image, multispectral

# Function to detect nanofiber diameters
def measure_nanofiber_diameter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val = threshold_otsu(blurred)
    binary = blurred > thresh_val
    labels = measure.label(binary)
    properties = measure.regionprops(labels)
    diameters = []
    for prop in properties:
        if prop.area > 50:
            min_diameter = prop.minor_axis_length
            max_diameter = prop.major_axis_length
            avg_diameter = (min_diameter + max_diameter) / 2
            diameters.append(avg_diameter)
    return diameters

# Function to highlight nanofibers
def highlight_nanofibers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    result = image.copy()
    result[np.where(edges != 0)] = [255, 0, 0]
    return result

# Function to measure distance between two points
def measure_distance(points, scale_nm_per_pixel):
    if len(points) == 2:
        p1, p2 = points
        distance_pixels = np.linalg.norm(np.array(p1) - np.array(p2))
        distance_nm = distance_pixels * scale_nm_per_pixel
        return distance_pixels, distance_nm
    return None, None

st.title("E-SPIN Nanofiber Analysis Tool")

uploaded_image = st.file_uploader("Upload a Nanofiber Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    binary, grayscale, color, multispectral = convert_image(img)

    st.subheader("Image Conversions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(binary, caption="Binary", use_column_width=True)
    with col2:
        st.image(grayscale, caption="Grayscale", use_column_width=True)
    with col3:
        st.image(color, caption="Color", use_column_width=True)
    with col4:
        st.image(multispectral, caption="Multispectral", use_column_width=True)

    diameters = measure_nanofiber_diameter(img)
    st.subheader("Nanofiber Diameters")
    df = pd.DataFrame({"Nanofiber #": list(range(1, len(diameters) + 1)), "Diameter (µm)": diameters})
    st.dataframe(df)
    fig = px.bar(df, x="Nanofiber #", y="Diameter (µm)", title="Nanofiber Diameter Distribution")
    st.plotly_chart(fig, use_container_width=True)

    highlighted_img = highlight_nanofibers(img)
    st.subheader("Highlighted Nanofibers")
    st.image(highlighted_img, channels="BGR", use_column_width=True, caption="Detected Nanofibers with Red Edges")

    st.subheader("Manual Nanofiber Diameter Measurement")
    scale_nm_per_pixel = st.number_input("Enter scale (nm per pixel):", min_value=0.1, max_value=1000.0, value=1.0)
    points = st.session_state.get("points", [])

    st.write("Click on two points on the image to measure the diameter:")
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def onclick(event):
        if len(points) < 2:
            points.append((event.xdata, event.ydata))
            st.session_state["points"] = points

    fig.canvas.mpl_connect("button_press_event", onclick)

    if len(points) == 2:
        p1, p2 = points
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "ro-")
        distance_pixels, distance_nm = measure_distance(points, scale_nm_per_pixel)
        st.write(f"Distance in pixels: {distance_pixels:.2f}")
        st.write(f"Estimated Nanofiber Diameter: {distance_nm:.2f} nm")
    st.pyplot(fig)

    if st.button("Reset Measurement"):
        st.session_state["points"] = []

if st.button("Reset All"):
    st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from skimage import measure
# from skimage.color import rgb2gray
# from skimage.filters import threshold_otsu
# import matplotlib.pyplot as plt

# # Function to convert image into different types
# def convert_image(image):
#     # Convert to grayscale
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Convert to binary
#     _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)

#     # Convert to multispectral (For simulation, split into 3 channels)
#     multispectral = cv2.merge([grayscale, binary, np.zeros_like(binary)])

#     return binary, grayscale, image, multispectral

# # Function to detect nanofiber diameters
# def measure_nanofiber_diameter(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Otsu thresholding
#     thresh_val = threshold_otsu(blurred)
#     binary = blurred > thresh_val

#     # Label connected components
#     labels = measure.label(binary)
#     properties = measure.regionprops(labels)

#     diameters = []
#     for prop in properties:
#         if prop.area > 50:  # Filtering noise
#             min_diameter = prop.minor_axis_length
#             max_diameter = prop.major_axis_length
#             avg_diameter = (min_diameter + max_diameter) / 2
#             diameters.append(avg_diameter)

#     return diameters

# # Function to highlight nanofiber edges
# def highlight_nanofibers(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)

#     result = image.copy()
#     result[np.where(edges != 0)] = [255, 0, 0]  # Highlight edges in red
#     return result

# # Function to measure distance between two points
# def measure_distance(points, scale_nm_per_pixel):
#     if len(points) == 2:
#         p1, p2 = points
#         distance_pixels = np.linalg.norm(np.array(p1) - np.array(p2))
#         distance_nm = distance_pixels * scale_nm_per_pixel
#         return distance_pixels, distance_nm
#     return None, None

# # Streamlit UI
# st.title("Nanofiber Analysis Tool")

# # Upload image
# uploaded_image = st.file_uploader("Upload a Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Convert image types
#     binary, grayscale, color, multispectral = convert_image(img)

#     # Display image conversions
#     st.subheader("Image Conversions")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.image(binary, caption="Binary", use_column_width=True)
#     with col2:
#         st.image(grayscale, caption="Grayscale", use_column_width=True)
#     with col3:
#         st.image(color, caption="Color", use_column_width=True)
#     with col4:
#         st.image(multispectral, caption="Multispectral", use_column_width=True)

#     # Nanofiber diameter measurement
#     diameters = measure_nanofiber_diameter(img)

#     # Display results in table
#     st.subheader("Nanofiber Diameters")
#     df = pd.DataFrame({"Nanofiber #": list(range(1, len(diameters) + 1)), "Diameter (µm)": diameters})
#     st.dataframe(df)

#     # Display diameter chart
#     fig = px.bar(df, x="Nanofiber #", y="Diameter (µm)", title="Nanofiber Diameter Distribution")
#     st.plotly_chart(fig, use_container_width=True)

#     # Apply edge highlighting
#     highlighted_img = highlight_nanofibers(img)

#     # Display edge-highlighted nanofibers
#     st.subheader("Highlighted Nanofibers")
#     st.image(highlighted_img, channels="BGR", use_column_width=True, caption="Detected Nanofibers with Red Edges")

#     # **New Feature: Distance Measurement**
#     st.subheader("Manual Nanofiber Diameter Measurement")
#     scale_nm_per_pixel = st.number_input("Enter scale (nm per pixel):", min_value=0.1, max_value=1000.0, value=1.0)

#     points = st.session_state.get("points", [])
    
#     # Image display for manual selection
#     st.write("Click on two points on the image to measure the diameter:")
#     fig, ax = plt.subplots()
#     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#     # Click handling
#     def onclick(event):
#         if len(points) < 2:
#             points.append((event.xdata, event.ydata))
#             st.session_state["points"] = points

#     fig.canvas.mpl_connect("button_press_event", onclick)

#     # Show points and draw line if both points exist
#     if len(points) == 2:
#         p1, p2 = points
#         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "ro-")

#         # Calculate distance
#         distance_pixels, distance_nm = measure_distance(points, scale_nm_per_pixel)
#         st.write(f"Distance in pixels: {distance_pixels:.2f}")
#         st.write(f"Estimated Nanofiber Diameter: {distance_nm:.2f} nm")

#     st.pyplot(fig)

#     # Reset button for manual selection
#     if st.button("Reset Measurement"):
#         st.session_state["points"] = []

# # Reset button
# if st.button("Reset All"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from skimage import measure
# from skimage.color import rgb2gray
# from skimage.filters import threshold_otsu

# # Function to convert image into different types
# def convert_image(image):
#     # Convert to grayscale
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Convert to binary
#     _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)

#     # Convert to multispectral (For simulation, split into 3 channels)
#     multispectral = cv2.merge([grayscale, binary, np.zeros_like(binary)])

#     return binary, grayscale, image, multispectral

# # Function to detect nanofiber diameters
# def measure_nanofiber_diameter(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Otsu thresholding
#     thresh_val = threshold_otsu(blurred)
#     binary = blurred > thresh_val

#     # Label connected components
#     labels = measure.label(binary)
#     properties = measure.regionprops(labels)

#     diameters = []
#     for prop in properties:
#         if prop.area > 50:  # Filtering noise
#             min_diameter = prop.minor_axis_length
#             max_diameter = prop.major_axis_length
#             avg_diameter = (min_diameter + max_diameter) / 2
#             diameters.append(avg_diameter)

#     return diameters

# # Function to highlight nanofiber edges
# def highlight_nanofibers(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)

#     result = image.copy()
#     result[np.where(edges != 0)] = [255, 0, 0]  # Highlight edges in red
#     return result

# # Streamlit UI
# st.title("Nanofiber Analysis Tool")

# # Upload image
# uploaded_image = st.file_uploader("Upload a Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Convert image types
#     binary, grayscale, color, multispectral = convert_image(img)

#     # Display image conversions
#     st.subheader("Image Conversions")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.image(binary, caption="Binary", use_column_width=True)
#     with col2:
#         st.image(grayscale, caption="Grayscale", use_column_width=True)
#     with col3:
#         st.image(color, caption="Color", use_column_width=True)
#     with col4:
#         st.image(multispectral, caption="Multispectral", use_column_width=True)

#     # Nanofiber diameter measurement
#     diameters = measure_nanofiber_diameter(img)

#     # Display results in table
#     st.subheader("Nanofiber Diameters")
#     df = pd.DataFrame({"Nanofiber #": list(range(1, len(diameters) + 1)), "Diameter (µm)": diameters})
#     st.dataframe(df)

#     # Display diameter chart
#     fig = px.bar(df, x="Nanofiber #", y="Diameter (µm)", title="Nanofiber Diameter Distribution")
#     st.plotly_chart(fig, use_container_width=True)

#     # Apply edge highlighting
#     highlighted_img = highlight_nanofibers(img)

#     # Display edge-highlighted nanofibers
#     st.subheader("Highlighted Nanofibers")
#     st.image(highlighted_img, channels="BGR", use_column_width=True, caption="Detected Nanofibers with Red Edges")

# # Reset button
# if st.button("Reset"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from sklearn.decomposition import PCA

# # Function to detect and measure nanofiber diameters while avoiding duplicates
# def measure_nanofiber_diameter(img):
#     # Convert to grayscale if needed
#     if len(img.shape) == 3:
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         img_gray = img

#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

#     # Apply edge detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Use morphological transformations to enhance nanofiber structure
#     kernel = np.ones((3, 3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     img_colored = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for marking
#     diameters = []
#     bounding_boxes = []
#     min_distance = 15  # Minimum pixel distance to avoid duplicate measurements

#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         diameter = max(w, h)  # Estimate diameter as max of width/height

#         # Avoid duplicate measurements by checking overlapping bounding boxes
#         if not any(abs(x - bx) < min_distance and abs(y - by) < min_distance for bx, by, _, _ in bounding_boxes):
#             diameters.append(diameter)
#             bounding_boxes.append((x, y, w, h))

#             # Draw bounding box & label the detected nanofiber
#             cv2.rectangle(img_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(img_colored, f"{diameter:.1f}px", (x, y - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return diameters, img_colored

# # Function to convert image into different types
# def convert_image_types(uploaded_image):
#     if uploaded_image is None:
#         return None, None, None, None, None

#     # Convert uploaded file to OpenCV format
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Convert to grayscale
#     grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Convert to binary (Thresholding)
#     _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

#     # Convert to multispectral using PCA (simulated for 3 channels)
#     img_float = img.astype(np.float32) / 255.0  # Normalize
#     reshaped_img = img_float.reshape((-1, 3))  # Flatten
#     pca = PCA(n_components=3)  # Reduce to 3 principal components
#     multispectral = pca.fit_transform(reshaped_img).reshape(img.shape)

#     return img, grayscale, binary, multispectral

# # Streamlit UI
# st.title("Nanofiber Analysis & Image Conversion Tool")

# uploaded_image = st.file_uploader("Upload Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     # Convert image into different types
#     color, grayscale, binary, multispectral = convert_image_types(uploaded_image)

#     # Display the different types of images
#     st.subheader("Original Image (Color)")
#     st.image(color, channels="BGR", use_column_width=True)

#     st.subheader("Grayscale Image")
#     st.image(grayscale, use_column_width=True, clamp=True)

#     st.subheader("Binary Image")
#     st.image(binary, use_column_width=True, clamp=True)

#     st.subheader("Multispectral Image (PCA Reduced)")
#     st.image(multispectral, use_column_width=True, clamp=True)

#     # Perform nanofiber measurement on the grayscale image
#     st.subheader("Nanofiber Diameter Measurement")
#     diameters, processed_img = measure_nanofiber_diameter(grayscale)

#     if diameters:
#         st.success(f"✅ Detected {len(diameters)} unique nanofibers.")

#         # Display detected diameters in a table
#         df = pd.DataFrame({"Nanofiber Index": range(1, len(diameters) + 1), "Diameter (pixels)": diameters})
#         st.dataframe(df, hide_index=True)

#         # Display chart using Plotly
#         fig = px.bar(df, x="Nanofiber Index", y="Diameter (pixels)", title="Nanofiber Diameter Distribution",
#                      labels={"Diameter (pixels)": "Diameter (pixels)", "Nanofiber Index": "Nanofiber"},
#                      text_auto=True)
#         st.plotly_chart(fig, use_container_width=True)

#         # Convert processed image for Streamlit display
#         processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#         st.image(processed_img_rgb, caption="Outlined & Measured Nanofibers", use_column_width=True)

#     else:
#         st.warning("No nanofibers detected. Try another image.")

# # Reset button
# if st.button("Reset"):
#     st.experimental_rerun()










# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px

# # Function to detect and measure nanofiber diameters while avoiding duplicates
# def measure_nanofiber_diameter(uploaded_image):
#     if uploaded_image is None:
#         st.warning("Please upload an image.")
#         return None, None, None

#     # Convert uploaded image to OpenCV format
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#     img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for marking

#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(img, (5, 5), 0)

#     # Apply edge detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Use morphological transformations to enhance nanofiber structure
#     kernel = np.ones((3, 3), np.uint8)
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     diameters = []
#     bounding_boxes = []
#     min_distance = 15  # Minimum pixel distance to avoid duplicate measurements

#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         diameter = max(w, h)  # Estimate diameter as max of width/height

#         # Avoid duplicate measurements by checking overlapping bounding boxes
#         if not any(abs(x - bx) < min_distance and abs(y - by) < min_distance for bx, by, _, _ in bounding_boxes):
#             diameters.append(diameter)
#             bounding_boxes.append((x, y, w, h))

#             # Draw bounding box & label the detected nanofiber
#             cv2.rectangle(img_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(img_colored, f"{diameter:.1f}px", (x, y - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return diameters, img_colored

# # Streamlit UI
# st.title("Nanofiber Diameter Measurement Tool")

# uploaded_image = st.file_uploader("Upload Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     diameters, processed_img = measure_nanofiber_diameter(uploaded_image)

#     if diameters:
#         st.success(f"✅ Detected {len(diameters)} unique nanofibers.")

#         # Display detected diameters in a table
#         df = pd.DataFrame({"Nanofiber Index": range(1, len(diameters) + 1), "Diameter (pixels)": diameters})
#         st.dataframe(df, hide_index=True)

#         # Display chart using Plotly
#         fig = px.bar(df, x="Nanofiber Index", y="Diameter (pixels)", title="Nanofiber Diameter Distribution",
#                      labels={"Diameter (pixels)": "Diameter (pixels)", "Nanofiber Index": "Nanofiber"},
#                      text_auto=True)
#         st.plotly_chart(fig, use_container_width=True)

#         # Convert processed image for Streamlit display
#         processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#         st.image(processed_img_rgb, caption="Outlined & Measured Nanofibers", use_column_width=True)

#     else:
#         st.warning("No nanofibers detected. Try another image.")

# # Reset button
# if st.button("Reset"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from skimage import measure

# # Function to measure nanofiber diameters and mark them on the image
# def measure_nanofiber_diameter(uploaded_image):
#     if uploaded_image is None:
#         st.warning("Please upload an image.")
#         return None, None, None

#     # Convert uploaded image to OpenCV format
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#     img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for marking

#     # Apply edge detection
#     edges = cv2.Canny(img, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     diameters = []
#     positions = []
#     min_distance = 10  # Minimum distance to avoid duplicate measurements

#     for cnt in contours:
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         diameter = 2 * radius

#         # Check for duplicate measurements
#         if all(abs(diameter - d) > min_distance for d in diameters):
#             diameters.append(diameter)
#             positions.append((int(x), int(y)))

#             # Draw circle around detected nanofiber
#             cv2.circle(img_colored, (int(x), int(y)), int(radius), (0, 255, 0), 2)
#             cv2.putText(img_colored, f"{diameter:.1f}px", (int(x) - 20, int(y) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return diameters, img_colored, positions

# # Streamlit UI
# st.title("Nanofiber Diameter Measurement Tool")

# uploaded_image = st.file_uploader("Upload Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     diameters, processed_img, positions = measure_nanofiber_diameter(uploaded_image)

#     if diameters:
#         st.success(f"✅ Detected {len(diameters)} unique nanofibers.")

#         # Display detected diameters in a table
#         df = pd.DataFrame({"Nanofiber Index": range(1, len(diameters) + 1), "Diameter (pixels)": diameters})
#         st.dataframe(df, hide_index=True)

#         # Display chart using Plotly
#         fig = px.bar(df, x="Nanofiber Index", y="Diameter (pixels)", title="Nanofiber Diameter Distribution",
#                      labels={"Diameter (pixels)": "Diameter (pixels)", "Nanofiber Index": "Nanofiber"},
#                      text_auto=True)
#         st.plotly_chart(fig, use_container_width=True)

#         # Convert processed image for Streamlit display
#         processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#         st.image(processed_img_rgb, caption="Measured Nanofibers", use_column_width=True)

#     else:
#         st.warning("No nanofibers detected. Try another image.")

# # Reset button
# if st.button("Reset"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from skimage import measure

# # Function to process image and measure nanofiber diameters
# def measure_nanofiber_diameter(uploaded_image):
#     if uploaded_image is None:
#         st.warning("Please upload an image.")
#         return None, None

#     # Convert uploaded image to OpenCV format
#     file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

#     # Apply edge detection
#     edges = cv2.Canny(img, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     diameters = []
#     min_distance = 10  # Minimum distance to avoid duplicate measurements

#     for cnt in contours:
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         diameter = 2 * radius

#         # Check for duplicate measurements
#         if all(abs(diameter - d) > min_distance for d in diameters):
#             diameters.append(diameter)

#     return diameters, img

# # Streamlit UI
# st.title("Nanofiber Diameter Measurement Tool")

# uploaded_image = st.file_uploader("Upload Nanofiber Image", type=["jpg", "png", "jpeg"])

# if uploaded_image:
#     diameters, processed_img = measure_nanofiber_diameter(uploaded_image)

#     if diameters:
#         st.success(f"✅ Detected {len(diameters)} unique nanofibers.")

#         # Display the detected diameters in a table
#         df = pd.DataFrame({"Nanofiber Index": range(1, len(diameters) + 1), "Diameter (pixels)": diameters})
#         st.dataframe(df, hide_index=True)

#         # Display a chart using Plotly
#         fig = px.bar(df, x="Nanofiber Index", y="Diameter (pixels)", title="Nanofiber Diameter Distribution",
#                      labels={"Diameter (pixels)": "Diameter (pixels)", "Nanofiber Index": "Nanofiber"},
#                      text_auto=True)
#         st.plotly_chart(fig, use_container_width=True)

#     else:
#         st.warning("No nanofibers detected. Try another image.")

# # Reset button
# if st.button("Reset"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from skimage import io, color, filters, measure

# # Function to process image and measure nanofiber diameters
# def measure_nanofiber_diameter(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Use edge detection
#     edges = cv2.Canny(blurred, 50, 150)
    
#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     diameters = []
    
#     for contour in contours:
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         diameter = 2 * radius
#         diameters.append(diameter)
    
#     return diameters

# # Streamlit UI
# st.title("🔬 Nanofiber Diameter Measurement Tool")
# st.write("Upload an image to measure the diameter of nanofibers.")

# # Upload image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Convert image file to OpenCV format
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Display the uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Process image
#     diameters = measure_nanofiber_diameter(image)

#     if diameters:
#         # Create DataFrame for results
#         df = pd.DataFrame({"Fiber #": range(1, len(diameters) + 1), "Diameter (µm)": diameters})
        
#         # Display Table
#         st.subheader("📊 Measured Diameters")
#         st.dataframe(df)

#         # Plot chart using Plotly
#         fig = px.histogram(df, x="Diameter (µm)", title="Nanofiber Diameter Distribution", 
#                            labels={"Diameter (µm)": "Diameter (µm)"}, nbins=20)
#         st.plotly_chart(fig)

#     else:
#         st.warning("No nanofibers detected. Try uploading a clearer image.")

# # Reset Button
# if st.button("🔄 Reset"):
#     st.experimental_rerun()









# import streamlit as st
# import cv2
# import numpy as np
# from skimage import io, color, measure, filters, morphology
# import pandas as pd
# import matplotlib.pyplot as plt

# def process_image(image):
#     """Convert image to grayscale, apply thresholding, and detect nanofibers."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = filters.sobel(gray)  # Edge detection
#     threshold = filters.threshold_otsu(edges)
#     binary = edges > threshold
#     cleaned = morphology.remove_small_objects(binary, min_size=50)  # Remove noise
#     labeled_fibers = measure.label(cleaned)
#     properties = measure.regionprops(labeled_fibers)

#     diameters = [prop.equivalent_diameter for prop in properties]
#     return diameters

# def main():
#     st.title("Nanofiber Diameter Measurement")
#     st.write("Upload an image of nanofibers to measure their diameters.")

#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
#     if uploaded_file:
#         image = io.imread(uploaded_file)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Process image and measure diameters
#         diameters = process_image(image)

#         if diameters:
#             # Display results in a table
#             df = pd.DataFrame({"Nanofiber Index": range(1, len(diameters) + 1), "Diameter (µm)": diameters})
#             st.write("### Measured Diameters")
#             st.dataframe(df)

#             # Display results as a histogram
#             st.write("### Diameter Distribution")
#             fig, ax = plt.subplots()
#             ax.hist(diameters, bins=15, color='skyblue', edgecolor='black')
#             ax.set_xlabel("Diameter (µm)")
#             ax.set_ylabel("Frequency")
#             ax.set_title("Histogram of Nanofiber Diameters")
#             st.pyplot(fig)
#         else:
#             st.error("No nanofibers detected. Try using a clearer image.")

# if __name__ == "__main__":
#     main()
