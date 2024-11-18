import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.graph_objects as go
from model import Autoencoder, VAE, BIGAN
import tensorflow_probability as tfp
import random
import json

min_val=-6.2808752
max_val=7.4021031
ae_threshold=0.0024469797
vae_upperbound=0.6856143474578857
vae_lowerbound=0.6788296699523926
bigan_threshold=0.68333566

if 'anomaly_count' not in st.session_state:
    st.session_state['anomaly_count'] = 0
    
@st.cache_resource
def load_model():
    AEmodel = Autoencoder()
    AEmodel.encoder = tf.keras.models.load_model("model/autoencoder-encoder.keras")
    AEmodel.decoder = tf.keras.models.load_model("model/autoencoder-decoder.keras")
    
    VAEmodel = VAE()
    VAEmodel.encoder = tf.keras.models.load_model("model/vae-encoder.keras")
    VAEmodel.decoder = tf.keras.models.load_model("model/vae-decoder.keras")
    
    BIGANmodel = BIGAN()
    BIGANmodel.generator = tf.keras.models.load_model("model/bigan-generator.keras")
    BIGANmodel.encoder = tf.keras.models.load_model("model/bigan-encoder.keras")
    BIGANmodel.discriminator = tf.keras.models.load_model("model/bigan-discriminator.keras")
    return AEmodel, VAEmodel, BIGANmodel
    # return VAEmodel

    
def process_drawing(drawing_data, num_points=140, y_min=-8, y_max=8):
    """Process the drawing data into evenly spaced points and scale them."""
    if not drawing_data["objects"]:
        return None
        
    # Extract line coordinates
    line = drawing_data["objects"][0]
    if "path" not in line:
        return None
        
    # Extract points from the path
    points = []
    for cmd in line["path"]:
        if len(cmd) > 1:  # Valid coordinate pair
            points.append([cmd[1], cmd[2]])
    
    if not points:
        return None
        
    # Convert to numpy array
    points = np.array(points)
    
    # Calculate cumulative distances
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Add starting point
    
    # Create evenly spaced points
    total_distance = distances[-1]
    even_distances = np.linspace(0, total_distance, num_points)
    
    # Interpolate to get evenly spaced points
    x_interp = np.interp(even_distances, distances, points[:, 0])
    y_interp = np.interp(even_distances, distances, points[:, 1])
    
    # Scale y values from canvas coordinates to -8 to 8 range
    # Invert the y-coordinates here (subtract from canvas height)
    canvas_height = 400  # Height of the canvas in pixels
    y_inverted = canvas_height - y_interp  # Invert Y coordinates
    y_scaled = ((y_inverted - canvas_height/2) / (canvas_height/2)) * 8
    
    return y_scaled

def process_csv(df):
    try:
        # df = pd.read_csv(file)
        # Get all rows but only first 140 columns
        data = df.iloc[:, :140].values
        if data.shape[1] != 140:
            st.error("CSV must contain at least 140 columns")
            return None
        return data
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def check_anomalies(row_data, models, thresholds):
    autoencoder, vae, bigan = models
    ae_threshold, vae_bounds, bigan_threshold = thresholds
    
    # Scale the data
    scaled_processed_data = (row_data - min_val) / (max_val - min_val)
    scaled_processed_data = tf.cast(scaled_processed_data, tf.float32)
    reconstructed_ae = autoencoder.predict(scaled_processed_data)
    reconstructed_vae = vae.predict(scaled_processed_data)[0]
    reconstructed_bigan = bigan.predict(scaled_processed_data)
    
    # Get predictions
    ae_pred = bool(tf.math.less(tf.keras.losses.mse(reconstructed_ae, scaled_processed_data), ae_threshold).numpy()[0])
    
    losses = vae.calculate_loss(scaled_processed_data)
    vae_pred = vae_lowerbound < losses.numpy()[0] < vae_upperbound
    
    bigan_pred = bool(tf.math.less(tf.keras.losses.binary_crossentropy(reconstructed_bigan, scaled_processed_data), bigan_threshold).numpy()[0])
    count = int(ae_pred) + int(vae_pred) + int(bigan_pred)
    if count < 2:
        st.markdown(f"### Row {idx}")
        # Create a visualization of the processed data and reconstructed data
        fig_ae = go.Figure()

        # Add input data trace
        fig_ae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))

        # Add reconstructed data trace
        fig_ae.add_trace(go.Scatter(
            y=reconstructed_ae[0],
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))

        # Update layout with legend inside the plot
        fig_ae.update_layout(
            height=400,
            width=600,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                x=0.02,  # Position of the legend on x-axis (0.0 is far left, 1.0 is far right)
                y=0.98,  # Position of the legend on y-axis (0.0 is bottom, 1.0 is top)
                xanchor="left",  # Align legend's x position
                yanchor="top",   # Align legend's y position
                # bgcolor="rgba(255,255,255,0.7)",  # Set a background color for better readability
                bordercolor="black",
                borderwidth=1
            )
        )
        # VAE Plot
        fig_vae = go.Figure()
        fig_vae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_vae.add_trace(go.Scatter(
            y=reconstructed_vae.reshape(140),
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))
        fig_vae.update_layout(
            legend=dict(
                x=0.02,  # Adjust x position
                y=0.98,  # Adjust y position
                xanchor="left",
                yanchor="top",
                bordercolor=None,  # No border color
                borderwidth=0      # No border width
            )
        )

        # BiGAN Plot
        fig_bigan = go.Figure()
        fig_bigan.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_bigan.add_trace(go.Scatter(
            y=reconstructed_bigan[0],
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))
        fig_bigan.update_layout(
            legend=dict(
                x=0.02,  # Adjust x position
                y=0.98,  # Adjust y position
                xanchor="left",
                yanchor="top",
                bordercolor=None,  # No border color
                borderwidth=0      # No border width
            )
        )

        # Create 3 columns for the plots
        col1, col2, col3 = st.columns(3)

        # Adjust common figure parameters for smaller width
        figure_params = dict(
            height=350,
            width=400,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )

        # Update all figures with common parameters
        fig_ae.update_layout(**figure_params)
        fig_vae.update_layout(**figure_params)
        fig_bigan.update_layout(**figure_params)

        # Column 1: AE
        with col1:
            st.plotly_chart(fig_ae, use_container_width=True)
            st.write("AE Predictions:", "Normal" if ae_pred else "Anomaly")
            st.write("AE Loss:", float(tf.keras.losses.mse(reconstructed_ae, scaled_processed_data)[0]))
            st.write("AE Threshold:", ae_threshold)

        # Column 2: VAE
        with col2:
            st.plotly_chart(fig_vae, use_container_width=True)
            st.write("VAE Predictions:", "Normal" if vae_pred else "Anomaly")
            st.write("VAE Loss:", losses.numpy()[0])
            st.write("VAE Upper/Lower:", vae_upperbound, '/', vae_lowerbound)

        # Column 3: BIGAN
        with col3:
            st.plotly_chart(fig_bigan, use_container_width=True)
            st.write("BIGAN Predictions:", "Normal" if bigan_pred else "Anomaly")
            st.write("BIGAN Loss:", float(tf.keras.losses.binary_crossentropy(bigan.predict(scaled_processed_data), scaled_processed_data)))
            st.write("BIGAN Threshold:", bigan_threshold)
    return {
        'ae_pred': ae_pred,
        'vae_pred': vae_pred,
        'bigan_pred': bigan_pred
        }

# Streamlit app
st.set_page_config(page_title="ECG Analysis", layout="wide")
st.markdown("<h1 style='text-align: center;'>Analyze ECG</h1>", unsafe_allow_html=True)

# Add custom CSS with more specific selectors
# Add custom CSS for layout
st.markdown("""
    <style>
    .canvas-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Create container for canvas section
with st.container():
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    
    # Create columns
    left_space, col1, col2, right_space = st.columns([0.5, 1.5, 1, 0.5])

    # Canvas in col1
    with col1:
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas"
        canvas_result = st_canvas(
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=400,
            width=600,
            display_toolbar=False,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.get('canvas_reset_count', 0)}",
        )

    # Buttons in col2
    with col2:
        title = st.write("<p style=\"font-size:19px\"> Draw something and see the magic happens 🗿</p>", unsafe_allow_html=True)
        delete_btn = st.button("Delete Drawing")
        analyze_btn = st.button("Analyze Drawing")
        upload_btn = st.file_uploader("Upload CSV", type=['csv'])
        row_amount_placeholder = st.empty()
        # Placeholder for the download button
        download_button_placeholder = st.empty()

    # Reset canvas on delete
    if delete_btn:
        st.session_state.canvas_key = f"canvas_{st.session_state.get('canvas_reset_count', 0)}"
        st.session_state.canvas_reset_count = st.session_state.get("canvas_reset_count", 0) + 1
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)

autoencoder, vae, bigan = load_model()
# Process the uploaded CSV file if a file is uploaded
if upload_btn is not None:
    try:
        df = pd.read_csv(upload_btn)
        max_value = len(df.index)
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
    row_amount = row_amount_placeholder.number_input(f"Number of rows to process (Max amount: {max_value})", min_value=1, max_value=max_value, value=None, step=1)
    if row_amount and analyze_btn is not None:
        processed_data = process_csv(df)[:row_amount]
        results = []
        if processed_data is not None:
            models = (autoencoder, vae, bigan)
            thresholds = (ae_threshold, (vae_lowerbound, vae_upperbound), bigan_threshold)
            total_rows = len(processed_data)

            # Reset anomaly count and create placeholders
            st.session_state['anomaly_count'] = 0
            anomaly_placeholder = st.empty()
            progress_bar = st.progress(0)

            for idx, row in enumerate(processed_data):
                row_data = row.reshape(1, -1)
                result = check_anomalies(row_data, models, thresholds)
                ae_pred = result['ae_pred']
                vae_pred = result['vae_pred']
                bigan_pred = result['bigan_pred']
                results.append({
                    'row_index': idx,
                    'ae_pred': "Normal" if ae_pred else "Anomaly",
                    'vae_pred': "Normal" if vae_pred else "Anomaly",
                    'bigan_pred': "Normal" if bigan_pred else "Anomaly"
                })
                count = int(ae_pred) + int(vae_pred) + int(bigan_pred)
                if count < 2:
                    st.session_state['anomaly_count'] += 1

                # Update the placeholder with the current anomaly count
                anomaly_placeholder.write(f"Found {st.session_state['anomaly_count']} anomalies in {total_rows} rows")

                # Update progress bar
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
            
            # Ensure progress bar completes
            progress_bar.progress(1.0)

            # Convert results to JSON string
            json_results = json.dumps(results, indent=4)

            # Update the download button placeholder to add the button below the upload button
            with col2:
                download_button_placeholder.download_button(
                    label="Download Results as JSON",
                    data=json_results,
                    file_name='results.json',
                    mime='application/json'
                )
elif (analyze_btn and canvas_result.json_data is not None):
    processed_data = process_drawing(canvas_result.json_data)
    if processed_data is not None:
        processed_data = np.array(processed_data).reshape(1, -1)
        scaled_processed_data = (processed_data - min_val) / (max_val - min_val)
        scaled_processed_data = tf.cast(scaled_processed_data, tf.float32)
        reconstructed_ae = autoencoder.predict(scaled_processed_data)
        reconstructed_vae = vae.predict(scaled_processed_data)[0]
        reconstructed_bigan = bigan.predict(scaled_processed_data)
        # print(reconstructed)
        
        ae_preds = tf.math.less(tf.keras.losses.mse(reconstructed_ae, scaled_processed_data), ae_threshold)
        print(ae_preds)
        
        losses = vae.calculate_loss(processed_data)
        percentile_lower = tfp.stats.percentile(losses, vae_lowerbound)
        percentile_upper = tfp.stats.percentile(losses, vae_upperbound)
        vae_preds = tf.logical_and(tf.greater_equal(losses, percentile_lower),tf.less_equal(losses, percentile_upper))
        vae_preds = tf.cast(vae_preds, dtype=tf.int32)
        # print(vae_preds)
        
        # bigan_preds = tf.math.less(tf.keras.losses.binary_crossentropy(reconstructed_bigan, processed_data), bigan_threshold) if bigan_threshold > np.mean(normal_train_loss) else tf.math.greater(tf.keras.losses.binary_crossentropy(reconstructed_bigan, processed_data), bigan_threshold)
        bigan_preds = tf.math.less(tf.keras.losses.binary_crossentropy(reconstructed_bigan, scaled_processed_data), bigan_threshold)
        print(bigan_preds)
        
        # Create a visualization of the processed data and reconstructed data
        fig_ae = go.Figure()

        # Add input data trace
        fig_ae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))

        # Add reconstructed data trace
        fig_ae.add_trace(go.Scatter(
            y=reconstructed_ae[0],
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))

        # Update layout with legend inside the plot
        fig_ae.update_layout(
            height=400,
            width=600,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                x=0.02,  # Position of the legend on x-axis (0.0 is far left, 1.0 is far right)
                y=0.98,  # Position of the legend on y-axis (0.0 is bottom, 1.0 is top)
                xanchor="left",  # Align legend's x position
                yanchor="top",   # Align legend's y position
                # bgcolor="rgba(255,255,255,0.7)",  # Set a background color for better readability
                bordercolor="black",
                borderwidth=1
            )
        )
        # VAE Plot
        fig_vae = go.Figure()
        fig_vae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_vae.add_trace(go.Scatter(
            y=reconstructed_vae.reshape(140),
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))
        fig_vae.update_layout(
            legend=dict(
                x=0.02,  # Adjust x position
                y=0.98,  # Adjust y position
                xanchor="left",
                yanchor="top",
                bordercolor=None,  # No border color
                borderwidth=0      # No border width
            )
        )

        # BiGAN Plot
        fig_bigan = go.Figure()
        fig_bigan.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_bigan.add_trace(go.Scatter(
            y=reconstructed_bigan[0],
            mode='lines',
            name='Reconstructed Data',
            line=dict(color='sandybrown')
        ))
        fig_bigan.update_layout(
            legend=dict(
                x=0.02,  # Adjust x position
                y=0.98,  # Adjust y position
                xanchor="left",
                yanchor="top",
                bordercolor=None,  # No border color
                borderwidth=0      # No border width
            )
        )

        # Create 3 columns for the plots
        col1, col2, col3 = st.columns(3)

        # Adjust common figure parameters for smaller width
        figure_params = dict(
            height=350,
            width=400,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )

        # Update all figures with common parameters
        fig_ae.update_layout(**figure_params)
        fig_vae.update_layout(**figure_params)
        fig_bigan.update_layout(**figure_params)

        # Column 1: AE
        with col1:
            st.plotly_chart(fig_ae, use_container_width=True)
            st.write("AE Predictions:", "Normal" if bool(ae_preds.numpy()[0]) else "Anomaly")
            st.write("AE Loss:", float(tf.keras.losses.mse(reconstructed_ae, scaled_processed_data)[0]))
            st.write("AE Threshold:", ae_threshold)

        # Column 2: VAE
        with col2:
            st.plotly_chart(fig_vae, use_container_width=True)
            st.write("VAE Predictions:", "Normal" if bool(vae_preds.numpy()[0]) else "Anomaly")
            st.write("VAE Loss:", float(vae.calculate_loss(processed_data)[0]))
            st.write("VAE Upper/Lower:", vae_upperbound, '/', vae_lowerbound)

        # Column 3: BIGAN
        with col3:
            st.plotly_chart(fig_bigan, use_container_width=True)
            st.write("BIGAN Predictions:", "Normal" if bool(bigan_preds.numpy()[0]) else "Anomaly")
            st.write("BIGAN Loss:", float(tf.keras.losses.binary_crossentropy(bigan.predict(scaled_processed_data), scaled_processed_data)))
            st.write("BIGAN Threshold:", bigan_threshold)

        # Display data statistics above the plots if needed
        # st.write("Min value of Input Data:", np.round(np.min(processed_data), 2))
        # st.write("Max value of Input Data:", np.round(np.max(processed_data), 2))
        # st.write("Min value of Scaled Input Data:", np.round(np.min(scaled_processed_data), 2))
        # st.write("Max value of Scaled Input Data:", np.round(np.max(scaled_processed_data), 2))
        
        # Option to download the data
        df = pd.DataFrame(processed_data[0], columns=['value'])
