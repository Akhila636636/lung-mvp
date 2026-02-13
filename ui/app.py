import streamlit as st
import plotly.graph_objects as go
import numpy as np
from inference.predict import predict
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, label
st.set_page_config(page_title="Lung CT Analysis", layout="wide")
st.title("🫁 Lung Nodule 3D Risk Prediction")
uploaded_file = st.file_uploader("Upload CT scan (.mhd)", type=["mhd"])
if uploaded_file:
    try:
        with st.spinner("Processing CT volume..."):
            prob, risk, volume, nodule_mask = predict(uploaded_file)
        
        # Validate prediction output
        if volume is None or volume.size == 0:
            st.error("Error: Could not process CT scan. Please upload a valid file.")
        else:
            st.metric("Risk Probability", f"{prob*100:.2f}%")
            st.write(risk)
            
            st.subheader("Interactive 3D Scan with Nodule Detection")
            
            # Downsample for smoother rendering
            vol_small = volume[::2, ::2, ::2]
            
            # Normalize safely for plotting
            vmin, vmax = vol_small.min(), vol_small.max()
            if vmax > vmin:
                vol_small = (vol_small - vmin) / (vmax - vmin)
            else:
                st.warning("Warning: Volume has uniform intensity values.")
            
            # Nodule detection settings
            st.sidebar.subheader("Nodule Detection Settings")
            
            # Use percentile-based threshold instead of absolute value
            percentile = st.sidebar.slider("Detection sensitivity (percentile)", 50, 99, 98, 
                                          help="Higher = detect only brightest regions (likely nodules)")
            min_nodule_size = st.sidebar.slider("Minimum nodule size (voxels)", 10, 500, 200,
                                               help="Larger = fewer false positives")
            max_nodule_size = st.sidebar.slider("Maximum nodule size (voxels)", 500, 5000, 1500,
                                               help="Smaller = fewer false positives")
            
            # Use percentile-based thresholding for better adaptive detection
            nodule_threshold = np.percentile(vol_small, percentile)
            
            # Apply Gaussian filter to enhance nodule-like structures (more aggressive)
            vol_filtered = gaussian_filter(vol_small, sigma=2.0)
            
            # Create binary mask for high-intensity regions (potential nodules)
            nodule_mask = vol_filtered > nodule_threshold
            
            # Apply stricter morphological operations to clean up the mask
            struct = np.ones((3, 3, 3))
            nodule_mask = binary_opening(nodule_mask, structure=struct)
            nodule_mask = binary_opening(nodule_mask, structure=struct)  # Apply twice for stricter filtering
            nodule_mask = binary_closing(nodule_mask, structure=struct)
            nodule_mask = binary_closing(nodule_mask, structure=struct)  # Apply twice
            
            # Label connected components (individual nodules)
            labeled_array, num_features = label(nodule_mask)
            
            # Filter by size range (removes noise and large artifacts)
            nodule_mask_filtered = np.zeros_like(labeled_array, dtype=bool)
            detected_nodules = []
            
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if min_nodule_size <= size <= max_nodule_size:
                    nodule_mask_filtered[labeled_array == i] = True
                    detected_nodules.append(size)
            
            num_nodules = len(detected_nodules)
            st.sidebar.success(f"✅ Detected {num_nodules} nodule(s)")
            
            if num_nodules == 0:
                st.info("No nodules detected at current settings. Try adjusting the sensitivity slider.")
            
            # Method 1: Mesh visualization with nodule highlighting
            try:
                from skimage import measure
                
                fig = go.Figure()
                
                # Extract isosurface for background (lung tissue) - exclude nodules
                threshold = 0.15
                vol_bg = vol_small.copy()
                vol_bg[nodule_mask_filtered] = 0  # Remove nodules from background
                
                verts_bg, faces_bg, _, _ = measure.marching_cubes(vol_bg, level=threshold)
                
                # Normalize vertex coordinates to [0, 1]
                if len(verts_bg) > 0:
                    verts_bg = verts_bg / np.array(vol_small.shape)
                    
                    fig.add_trace(go.Mesh3d(
                        x=verts_bg[:, 0],
                        y=verts_bg[:, 1],
                        z=verts_bg[:, 2],
                        i=faces_bg[:, 0],
                        j=faces_bg[:, 1],
                        k=faces_bg[:, 2],
                        opacity=0.4,
                        color='rgba(100, 150, 255, 0.4)',
                        name="Lung Tissue",
                        showlegend=True
                    ))
                
                # Extract isosurface for nodules (highlighted in red)
                if num_nodules > 0:
                    verts_nodules, faces_nodules, _, _ = measure.marching_cubes(
                        nodule_mask_filtered.astype(float), level=0.5
                    )
                    
                    if len(verts_nodules) > 0:
                        verts_nodules = verts_nodules / np.array(vol_small.shape)
                        
                        fig.add_trace(go.Mesh3d(
                            x=verts_nodules[:, 0],
                            y=verts_nodules[:, 1],
                            z=verts_nodules[:, 2],
                            i=faces_nodules[:, 0],
                            j=faces_nodules[:, 1],
                            k=faces_nodules[:, 2],
                            opacity=0.95,
                            color='rgba(255, 50, 50, 0.9)',
                            name="Detected Nodules",
                            showlegend=True,
                            hovertemplate='<b>Nodule</b><extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f"3D CT Volume with Nodule Highlighting ({num_nodules} nodules detected)",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode="data"
                    ),
                    height=700,
                    showlegend=True,
                    hovermode='closest',
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("scikit-image not installed. Using slice viewer instead.")
                
                # Fallback to slice viewer with nodule overlay
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
                    nodule_overlay = nodule_mask_filtered[slice_idx, :, :]
                elif axis == 1:
                    slice_data = vol_small[:, slice_idx, :]
                    nodule_overlay = nodule_mask_filtered[:, slice_idx, :]
                else:
                    slice_data = vol_small[:, :, slice_idx]
                    nodule_overlay = nodule_mask_filtered[:, :, slice_idx]
                
                # Create figure with base CT scan
                fig = go.Figure()
                
                fig.add_trace(go.Heatmap(
                    z=slice_data,
                    colorscale='Greys',
                    name="CT Scan",
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
                ))
                
                # Overlay nodule mask in red
                if num_nodules > 0:
                    fig.add_trace(go.Heatmap(
                        z=nodule_overlay.astype(float),
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(255,0,0,0.7)']],
                        name="Nodules",
                        showscale=False,
                        hovertemplate='Nodule: %{z}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"CT Slice {slice_idx} - Nodules in Red ({num_nodules} detected)",
                    height=600,
                    xaxis_title="X",
                    yaxis_title="Y",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display nodule statistics
            if num_nodules > 0:
                with st.expander("📊 Nodule Statistics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Nodules Detected", num_nodules)
                    
                    with col2:
                        st.metric("Total Nodule Voxels", int(np.sum(nodule_mask_filtered)))
                    
                    with col3:
                        st.metric("Avg Nodule Size", f"{np.mean(detected_nodules):.0f} voxels")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Largest Nodule", f"{np.max(detected_nodules)} voxels")
                    
                    with col2:
                        st.metric("Smallest Nodule", f"{np.min(detected_nodules)} voxels")
                    
                    with col3:
                        st.metric("Detection Threshold", f"{nodule_threshold:.3f}")
                    
                    st.bar_chart({"Nodule Size Distribution": detected_nodules})
            else:
                st.info("💡 No nodules detected. This may indicate a healthy scan or require threshold adjustment.")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure the file is a valid CT scan in .mhd format.")