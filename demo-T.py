import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.graph_objects as go
from model import Autoencoder, VAE, BIGAN
# import tensorflow_probability as tfp


def load_model():
    AEmodel = Autoencoder()
    AEmodel.encoder = tf.keras.models.load_model("autoencoder-encoder.keras")
    AEmodel.decoder = tf.keras.models.load_model("autoencoder-decoder.keras")
    
    VAEmodel = VAE()
    VAEmodel.encoder = tf.keras.models.load_model("vae-encoder.keras")
    VAEmodel.decoder = tf.keras.models.load_model("vae-decoder.keras")
    
    BIGANmodel = BIGAN()
    BIGANmodel.generator = tf.keras.models.load_model("bigan-generator.keras")
    BIGANmodel.encoder = tf.keras.models.load_model("bigan-encoder.keras")
    BIGANmodel.discriminator = tf.keras.models.load_model("bigan-discriminator.keras")
    st.write("Model weights after loading:", AEmodel.get_weights())
    st.write("Model weights after loading:", VAEmodel.get_weights())
    st.write("Model weights after loading:", BIGANmodel.get_weights())
    return AEmodel, VAEmodel, BIGANmodel
    # return VAEmodel
# Load pre-trained weights

# autoencoder = tf.keras.models.load_model('autoencoder.h5',compile=False)

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

# Streamlit app
st.title("Draw a Time Series")

# Create canvas with the same dimensions as the grid
canvas_result = st_canvas(
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    height=400,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.json_data is not None:
    # Process the drawing
    processed_data = process_drawing(canvas_result.json_data)
    autoencoder, vae, bigan = load_model()
    if processed_data is not None:
        st.write("Processed Data Points:", processed_data.shape)
        processed_data = np.array(processed_data).reshape(1, -1)
        min_val=-6.2808752
        max_val=7.4021031
        ae_threshold=0.0024469797
        vae_upperbound=55
        vae_lowerbound=0
        bigan_threshold=0.6088931560516357
        scaled_processed_data = (processed_data - min_val) / (max_val - min_val)
        scaled_processed_data = tf.cast(scaled_processed_data, tf.float32)
        reconstructed_ae = autoencoder.predict(scaled_processed_data)
        reconstructed_vae = vae.predict(scaled_processed_data)[0]
        reconstructed_bigan = bigan.predict(scaled_processed_data)
        # print(reconstructed)
        
        ae_preds = tf.math.less(tf.keras.losses.mse(reconstructed_ae, processed_data), ae_threshold)
        print(ae_preds)
        
        losses = vae.calculate_loss(processed_data)
        percentile_lower = tf.experimental.numpy.percentile(losses, vae_lowerbound)
        percentile_upper = tf.experimental.numpy.percentile(losses, vae_upperbound)
        vae_preds = tf.logical_and(tf.greater_equal(losses, percentile_lower), tf.less_equal(losses, percentile_upper))
        vae_preds = tf.cast(vae_preds, dtype=tf.int32)
        # print(vae_preds)
        
        # bigan_preds = tf.math.less(tf.keras.losses.binary_crossentropy(reconstructed_bigan, processed_data), bigan_threshold) if bigan_threshold > np.mean(normal_train_loss) else tf.math.greater(tf.keras.losses.binary_crossentropy(reconstructed_bigan, processed_data), bigan_threshold)
        bigan_preds = tf.math.less(tf.keras.losses.binary_crossentropy(reconstructed_bigan, processed_data), bigan_threshold)
        print(bigan_preds)
        
        # Create a visualization of the processed data and reconstructed data
        fig_ae = go.Figure()
        fig_ae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_ae.add_trace(go.Scatter(
            y=reconstructed_ae[0],
            mode='lines',
            name='Reconstructed Data'
        ))
        fig_ae.update_layout(
            height=400,
            width=600,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        fig_vae = go.Figure()
        fig_vae.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_vae.add_trace(go.Scatter(
            y=reconstructed_vae.reshape(140),
            mode='lines',
            name='Reconstructed Data'
        ))
        fig_vae.update_layout(
            height=400,
            width=600,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        fig_bigan = go.Figure()
        fig_bigan.add_trace(go.Scatter(
            y=scaled_processed_data[0],
            mode='lines',
            name='Input Data'
        ))
        fig_bigan.add_trace(go.Scatter(
            y=reconstructed_bigan[0],
            mode='lines',
            name='Reconstructed Data'
        ))
        fig_bigan.update_layout(
            height=400,
            width=600,
            yaxis_range=[-1, 1],
            xaxis_range=[0, 140],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        st.write("Min value of Input Data:", np.round(np.min(processed_data), 2))
        st.write("Max value of Input Data:", np.round(np.max(processed_data), 2))
        st.write("Min value of Scaled Input Data:", np.round(np.min(scaled_processed_data), 2))
        st.write("Max value of Scaled Input Data:", np.round(np.max(scaled_processed_data), 2))
        
        st.plotly_chart(fig_ae, use_container_width=True)
        st.write("AE Predictions:", ae_preds)
        st.write("AE Loss:", tf.keras.losses.mse(reconstructed_ae, processed_data))
        st.write("AE Threshold:", ae_threshold)
        
        st.plotly_chart(fig_vae, use_container_width=True)
        st.write("VAE Predictions:", vae_preds)
        st.write("VAE Loss:", vae.calculate_loss(processed_data))
        st.write("VAE Upper/Lower:", vae_upperbound, '/', vae_lowerbound)  
        
        st.plotly_chart(fig_bigan, use_container_width=True)
        st.write("BIGAN Predictions:", bigan_preds)
        st.write("BIGAN Loss:", bigan.calculate_loss(processed_data))
        st.write("BIGAN Threshold:", bigan_threshold)
        # # Show statistics
        # st.write("Processed Data Points:", processed_data[0])
        # st.write("Reconstructed Data Points:", reconstructed[0])

        
        # Option to download the data
        df = pd.DataFrame(processed_data[0], columns=['value'])
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='drawn_timeseries.csv',
            mime='text/csv',
        )
        df = pd.DataFrame(reconstructed[0], columns=['value'])
        st.download_button(
            label="Download data another as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='drawn_timeseries.csv',
            mime='text/csv',
        )