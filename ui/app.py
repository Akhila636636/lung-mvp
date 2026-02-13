import streamlit as st
import plotly.graph_objects as go
import numpy as np
from inference.predict import predict
st.set_page_config(page_title="Lung CT Analysis", layout="wide")
st.title("🫁 Lung Nodule 3D Risk Prediction")
uploaded_file = st.file_uploader("Upload CT scan (.mhd)", type=["mhd"])
if uploaded_file:
    try:
        with st.spinner("Processing CT volume..."):
            prob, risk, volume = predict(uploaded_file)
        
        # Validate prediction output
        if volume is None or volume.size == 0:
            st.error("Error: Could not process CT scan. Please upload a valid file.")
        else:
            st.metric("Risk Probability", f"{prob*100:.2f}%")
            st.write(risk)
            
            st.subheader("Interactive 3D Scan")
            
            # Downsample for smoother rendering
            vol_small = volume[::2, ::2, ::2]
            
            # Normalize safely for plotting
            vmin, vmax = vol_small.min(), vol_small.max()
            if vmax > vmin:
                vol_small = (vol_small - vmin) / (vmax - vmin)
            else:
                st.warning("Warning: Volume has uniform intensity values.")
            
            # Method 1: Isosurface extraction (more reliable)
            try:
                from skimage import measure
                
                # Extract isosurface at threshold
                threshold = 0.2
                verts, faces, _, _ = measure.marching_cubes(vol_small, level=threshold)
                
                # Normalize vertex coordinates to [0, 1]
                verts = verts / np.array(vol_small.shape)
                
                fig = go.Figure(data=[go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.7,
                    color='rgba(100, 150, 255, 0.5)',
                )])
                
                fig.update_layout(
                    title="3D CT Volume (Isosurface)",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode="data"
                    ),
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("scikit-image not installed. Using slice viewer instead.")
                # Fallback to slice viewer
                slice_axis = st.selectbox("View slices along axis:", ["Z (axial)", "Y (coronal)", "X (sagittal)"])
                
                if slice_axis == "Z (axial)":
                    slices = vol_small.shape[0]
                    axis = 0
                elif slice_axis == "Y (coronal)":
                    slices = vol_small.shape[1]
                    axis = 1
                else:
                    slices = vol_small.shape[2]
                    axis = 2
                
                slice_idx = st.slider("Slice number", 0, slices - 1, slices // 2)
                
                if axis == 0:
                    slice_data = vol_small[slice_idx, :, :]
                elif axis == 1:
                    slice_data = vol_small[:, slice_idx, :]
                else:
                    slice_data = vol_small[:, :, slice_idx]
                
                fig = go.Figure(data=go.Heatmap(z=slice_data, colorscale='Greys'))
                fig.update_layout(
                    title=f"CT Slice {slice_idx}",
                    height=600,
                    xaxis_title="X",
                    yaxis_title="Y"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure the file is a valid CT scan in .mhd format.")