import streamlit as st
import cv2
import tempfile
import os
import time
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from twilio.rest import Client
from geopy.geocoders import Nominatim
import piexif
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to extract GPS metadata (unchanged)
def get_geotags(image_path):
    """Extract GPS metadata (latitude & longitude) from an image."""
    try:
        img = Image.open(image_path)
        if "exif" not in img.info:
            return None, None, "No EXIF metadata found."
        
        exif_data = piexif.load(img.info['exif'])
        gps_info = exif_data.get("GPS", {})
        
        if not gps_info:
            return None, None, "No GPS data found."

        def convert_to_degrees(value):
            """Convert GPS coordinates from EXIF format to decimal degrees."""
            d, m, s = value
            return d[0] / d[1] + (m[0] / m[1]) / 60 + (s[0] / s[1]) / 3600

        if piexif.GPSIFD.GPSLatitude in gps_info and piexif.GPSIFD.GPSLongitude in gps_info:
            lat = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLatitude])
            lon = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLongitude])
            
            lat_ref = gps_info[piexif.GPSIFD.GPSLatitudeRef].decode()
            lon_ref = gps_info[piexif.GPSIFD.GPSLongitudeRef].decode()
            
            if lat_ref != "N":
                lat = -lat
            if lon_ref != "E":
                lon = -lon
            
            return lat, lon, None
        return None, None, "No GPS coordinates found."
    except Exception as e:
        return None, None, f"Error extracting GPS data: {str(e)}"

# Function to convert coordinates to location name (unchanged)
def get_location_name(latitude, longitude):
    """Convert latitude & longitude into a location name using OpenStreetMap."""
    try:
        geolocator = Nominatim(user_agent="accident_detection_app")
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        return location.address if location else "Location not found"
    except Exception as e:
        return f"Error getting location: {str(e)}"

# Function to send an alert with location information (unchanged)
def send_alert(image_path=None):
    TWILIO_ACCOUNT_SID = "AC2e3662b921379654f19a7fdb8d608f3b"
    TWILIO_AUTH_TOKEN = "c32e759f9c631d81a2ca9f09394bc5bf"
    TWILIO_PHONE_NUMBER = "+17174524635"
    RECIPIENT_PHONE_NUMBER = "+916302210655"

    location_info = "Location data unavailable"
    if image_path:
        latitude, longitude, error = get_geotags(image_path)
        if error:
            location_info = error
        elif latitude and longitude:
            location_info = get_location_name(latitude, longitude)
            location_info = f"Location: {location_info}\nCoordinates: ({latitude:.6f}, {longitude:.6f})"

    alert_message = f"\U0001F6A8 Alert: An accident has been detected!\n{location_info}"

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=alert_message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        logger.info(f"Alert sent successfully! Message SID: {message.sid}")
        st.success(f"\U0001F4E9 Alert sent! Message SID: {message.sid}")
    except Exception as e:
        error_msg = f"Failed to send alert: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        # Additional debug info
        logger.error(f"Twilio SID: {TWILIO_ACCOUNT_SID}")
        logger.error(f"Twilio Phone: {TWILIO_PHONE_NUMBER}")
        logger.error(f"Recipient Phone: {RECIPIENT_PHONE_NUMBER}")
    
    return location_info

# Function to set page style (unchanged)
def set_page_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
        }
        .stFileUploader>div>div>button {
            background-color: #2196F3;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Login check (unchanged)
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FF5733;'>\U0001F510 Login Page</h1>", unsafe_allow_html=True)
        username = st.text_input("\U0001F464 Username")
        password = st.text_input("\U0001F511 Password", type="password")
        if st.button("Login ✅"):
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try again.")

# Main function
def main():
    set_page_style()
    check_login()
    if not st.session_state.logged_in:
        return
    
    model = YOLO(r'C:\Users\Tharun\OneDrive\Desktop\loc\FrontEnd\best.pt')
    
    st.markdown("<h1 style='text-align: center; color: #008CBA;'>\U0001F697 YOLO Object Detection</h1>", unsafe_allow_html=True)
    st.write("📤 Upload an image, video, or use the camera for real-time detection.")
    
    option = st.radio("Choose Input Source:", ("Upload Image or Video", "Live Camera"))
    
    if option == "Upload Image or Video":
        uploaded_file = st.file_uploader("📂 Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        if uploaded_file is not None:
            file_type = uploaded_file.type
            accident_detected = False

            if "image" in file_type:
                # Save the uploaded file temporarily with its original EXIF data
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()

                # Process the image with YOLO
                results = model(temp_file.name)
                res_plotted = Image.fromarray(results[0].plot())
    
                detected_classes = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
                class_names = [results[0].names[int(cls_id)] for cls_id in detected_classes]
                logger.info(f"Detected Classes: {class_names}")
                accident_detected = any("accident" in name.lower() for name in class_names)

                st.image(res_plotted, caption="🎯 Detected Objects", use_container_width=True)
                st.success("✅ Image detection completed!")

                if accident_detected:
                    st.error("\U0001F6A8 Alert: Accident detected in the image!")
                    location_info = send_alert(temp_file.name)
                    st.info(f"📍 {location_info}")

                # Clean up temporary file
                os.unlink(temp_file.name)

            elif "video" in file_type:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_video_path = temp_file.name
                cap = cv2.VideoCapture(temp_video_path)
                width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS) or 30
                output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                with st.spinner("🔄 Processing the video... Please wait."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = model(frame)
                        frame_out = results[0].plot()
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                        out.write(frame_out)
                        for result in results:
                            if "accident" in result.names.values():
                                accident_detected = True

                cap.release()
                out.release()

                if accident_detected:
                    st.error("\U0001F6A8 Alert: Accident detected in the video!")
                    location_info = send_alert()  # No location for video
                    st.info(f"📍 {location_info}")

                if os.path.exists(output_video_path):
                    st.success("🎉 Video detection completed successfully!")
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=open(output_video_path, "rb").read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

    elif option == "Live Camera":
        model = YOLO(r'C:\Users\Tharun\OneDrive\Desktop\loc\FrontEnd\best.pt')
        st.warning("📹 Capturing live video from webcam...")

        st.title("🚗 Real-time Accident Detection using YOLO")
        if "recording" not in st.session_state:
            st.session_state.recording = False
        if "video_path" not in st.session_state:
            st.session_state.video_path = None
        if "accident_detected" not in st.session_state:
            st.session_state.accident_detected = False

        if st.button("🎥 Start Recording"):
            st.session_state.accident_detected = False
            st.warning("📹 Recording for 5 seconds...")
    
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Failed to access webcam.")
                return

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video.name, fourcc, 20.0, (frame_width, frame_height))

            frame_placeholder = st.empty()

            start_time = time.time()
            while time.time() - start_time < 5: 
                ret, frame = cap.read()     
                if not ret:
                    break
                out.write(frame)
                frame_placeholder.image(frame, channels="BGR", caption="📹 Recording...")

            # Modified: Explicitly release resources before closing
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            st.session_state.video_path = temp_video.name
            st.success("📌 Recording stopped.")
            temp_video.close()  # Ensure the file is closed

        if st.session_state.video_path and not st.session_state.recording:
            st.subheader("🔍 Processing Video for Accident Detection...")
            cap = cv2.VideoCapture(st.session_state.video_path)
            if not cap.isOpened():
                st.error("❌ Failed to open recorded video.")
                return

            accident_detected = False
            best_confidence = 0.0
            best_frame = None
            best_frame_path = None

            # Modified: Track the frame with highest confidence and save it for EXIF extraction
            with st.spinner("🔄 Processing video frames... Please wait."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    frame_out = results[0].plot()

                    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
                        detected_classes = results[0].boxes.cls.tolist()
                        confidences = results[0].boxes.conf.tolist()
                        class_names = [results[0].names[int(cls_id)] for cls_id in detected_classes]

                        for i, name in enumerate(class_names):
                            if "accident" in name.lower() and confidences[i] > best_confidence:
                                accident_detected = True
                                best_confidence = confidences[i]
                                best_frame = frame_out
                                # Save frame temporarily to extract EXIF data later
                                temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                                cv2.imwrite(temp_frame_path, frame)
                                # Clean up previous best frame if exists
                                if best_frame_path and os.path.exists(best_frame_path):
                                    try:
                                        os.unlink(best_frame_path)
                                    except PermissionError as e:
                                        logger.warning(f"Could not delete previous frame {best_frame_path}: {e}")
                                best_frame_path = temp_frame_path

            # Modified: Explicitly release video capture
            cap.release()

            # Modified: Display only the most confident frame and send a single alert
            if accident_detected and best_frame is not None and best_frame_path:
                st.error(f"🚨 Alert: Accident detected with confidence {best_confidence:.2f}!")
                st.image(best_frame, channels="RGB", caption="🚨 Most Confident Accident Frame")
                st.session_state.accident_detected = True
                location_info = send_alert(best_frame_path)
                st.info(f"📍 {location_info}")
                # Clean up the best frame file
                try:
                    os.unlink(best_frame_path)
                except PermissionError as e:
                    logger.warning(f"Could not delete frame {best_frame_path}: {e}")

            else:
                st.success("✅ No accident detected.")

            # Modified: Clean up temporary video with error handling
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                try:
                    os.unlink(st.session_state.video_path)
                except PermissionError as e:
                    logger.warning(f"Could not delete video {st.session_state.video_path}: {e}")
                    st.warning(f"⚠️ Temporary video file could not be deleted due to a permission issue. You may need to delete it manually: {st.session_state.video_path}")
            st.session_state.video_path = None

if __name__ == "__main__":
    main()