#!/usr/bin/env python3
"""
Streamlit UI for Person Re-Identification System
Simple and clean interface for all operations - API Client Version
"""

import streamlit as st
import os
import requests
import time
from pathlib import Path
from datetime import datetime

# API endpoints from environment variables
EXTRACT_API_URL = os.getenv("EXTRACT_API_URL", "http://localhost:8001")
REGISTER_API_URL = os.getenv("REGISTER_API_URL", "http://localhost:8002")
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8003")

# Page config
st.set_page_config(
    page_title="Person ReID System",
    page_icon="🎥",
    layout="wide"
)

# Title
st.title("🎥 Person Re-Identification System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Operation",
    ["📹 Extract Objects", "👤 Register Person", "🔍 Detect & Track", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: EXTRACT OBJECTS
# ============================================================================
if page == "📹 Extract Objects":
    st.header("📹 Extract Individual Objects from Video")
    st.markdown("Extract separate videos for each tracked person from multi-person footage")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mkv', 'mov'])
    
    with col2:
        st.markdown("### Parameters")
        model_type = st.selectbox("Model", ["mot17", "yolox"], index=0)
        min_frames = st.number_input("Min Frames", min_value=1, value=10, help="Minimum frames to save object")
        padding = st.number_input("Padding (px)", min_value=0, value=10, help="Padding around bbox")
        conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        track_thresh = st.slider("Track Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("🚀 Extract Objects", type="primary"):
        if video_file is None:
            st.error("Please upload a video file")
        else:
            with st.spinner("Uploading video and starting extraction..."):
                try:
                    # Prepare files and data for API call
                    files = {"video": (video_file.name, video_file.getvalue(), "video/mp4")}
                    data = {
                        "model_type": model_type,
                        "padding": padding,
                        "conf_thresh": conf_thresh,
                        "track_thresh": track_thresh,
                        "min_frames": min_frames
                    }

                    # Call Extract API
                    response = requests.post(f"{EXTRACT_API_URL}/extract", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]

                        # Store job_id in session state
                        st.session_state['extract_current_job_id'] = job_id

                        st.info(f"Job ID: {job_id}")

                        # Poll for status
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while True:
                            status_response = requests.get(f"{EXTRACT_API_URL}/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()
                                status_text.text(f"Status: {status['status']}")

                                if status["status"] == "completed":
                                    progress_bar.progress(100)
                                    st.success("✅ Extraction complete!")

                                    # Get results and cache in session state (for display outside polling loop)
                                    results_cache_key = f"extract_results_{job_id}"
                                    if results_cache_key not in st.session_state:
                                        results_response = requests.get(f"{EXTRACT_API_URL}/results/{job_id}")
                                        if results_response.status_code == 200:
                                            st.session_state[results_cache_key] = results_response.json()
                                            results = results_response.json()
                                        else:
                                            results = None
                                    else:
                                        results = st.session_state.get(results_cache_key)

                                    # Cache all download data (preview, files, log)
                                    if results and results['files']:
                                        # Cache all files (including preview)
                                        for idx, filename in enumerate(results['files']):
                                            cache_key = f"extract_file_{job_id}_{filename}"
                                            if cache_key not in st.session_state:
                                                try:
                                                    file_url = f"{EXTRACT_API_URL}/download/{job_id}/{filename}"
                                                    file_response = requests.get(file_url)
                                                    if file_response.status_code == 200:
                                                        st.session_state[cache_key] = file_response.content
                                                        # Also cache first file as preview
                                                        if idx == 0:
                                                            st.session_state[f"extract_preview_{job_id}"] = file_response.content
                                                except Exception as e:
                                                    pass

                                        # Cache log
                                        cache_key = f"extract_log_{job_id}"
                                        if cache_key not in st.session_state:
                                            try:
                                                log_url = f"{EXTRACT_API_URL}/download/log/{job_id}"
                                                log_response = requests.get(log_url)
                                                if log_response.status_code == 200:
                                                    st.session_state[cache_key] = log_response.content
                                            except Exception as e:
                                                pass

                                    break

                                elif status["status"] == "failed":
                                    st.error(f"❌ Extraction failed: {status.get('error', 'Unknown error')}")
                                    break

                                elif status["status"] == "processing":
                                    progress_bar.progress(50)

                            time.sleep(2)
                    else:
                        st.error(f"Failed to start extraction: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display results if available (after rerun from download button click)
    if 'extract_current_job_id' in st.session_state:
        job_id = st.session_state['extract_current_job_id']
        results_cache_key = f"extract_results_{job_id}"

        if results_cache_key in st.session_state:
            results = st.session_state[results_cache_key]

            st.markdown("---")
            st.markdown(f"### 📊 Extraction Results (Job: {job_id[:8]}...)")
            st.info(f"Found **{results['total_objects']}** objects")

            # Show list of extracted objects
            st.markdown("#### 📹 Extracted Object Videos:")
            for filename in results['files']:
                st.text(f"  • {filename}")

            # Preview first object video (if available)
            if results['files']:
                st.markdown("---")
                st.markdown("### 🎬 Preview (First Object)")
                first_file = results['files'][0]

                try:
                    # Fetch first object video for preview ONCE and cache
                    cache_key = f"extract_preview_{job_id}"
                    if cache_key not in st.session_state:
                        preview_url = f"{EXTRACT_API_URL}/download/{job_id}/{first_file}"
                        preview_response = requests.get(preview_url)
                        if preview_response.status_code == 200:
                            st.session_state[cache_key] = preview_response.content

                    if cache_key in st.session_state:
                        # Encode to base64 for inline display
                        import base64
                        video_base64 = base64.b64encode(st.session_state[cache_key]).decode()

                        video_html = f"""
                        <video width="100%" controls>
                            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)
                        st.caption(f"Preview: {first_file}")
                except Exception as e:
                    st.warning(f"Could not load preview: {e}")

            # Download all as ZIP - CACHE DATA
            st.markdown("---")
            st.markdown("### 📦 Download All Objects as ZIP")

            cache_key = f"extract_zip_{job_id}"

            # Fetch ZIP only if not cached
            if cache_key not in st.session_state:
                try:
                    zip_url = f"{EXTRACT_API_URL}/download/zip/{job_id}"
                    zip_response = requests.get(zip_url)
                    if zip_response.status_code == 200:
                        st.session_state[cache_key] = zip_response.content
                except Exception as e:
                    st.error(f"Failed to fetch ZIP: {e}")

            # Show download button with cached data
            if cache_key in st.session_state:
                st.download_button(
                    label="📦 Download All (ZIP)",
                    data=st.session_state[cache_key],
                    file_name=f"{job_id}_extracted_objects.zip",
                    mime="application/zip",
                    key=f"download_zip_{job_id}"
                )

            # Download log - CACHE DATA
            st.markdown("---")
            cache_key = f"extract_log_{job_id}"

            # Fetch only if not cached
            if cache_key not in st.session_state:
                try:
                    log_url = f"{EXTRACT_API_URL}/download/log/{job_id}"
                    log_response = requests.get(log_url)
                    if log_response.status_code == 200:
                        st.session_state[cache_key] = log_response.content
                except Exception as e:
                    st.error(f"Failed to fetch log: {e}")

            # Show download button with cached data
            if cache_key in st.session_state:
                st.download_button(
                    label="📄 Download Extraction Log",
                    data=st.session_state[cache_key],
                    file_name=f"{job_id}_extraction.log",
                    mime="text/plain",
                    key=f"download_log_{job_id}"
                )

# ============================================================================
# PAGE 2: REGISTER PERSON
# ============================================================================
elif page == "👤 Register Person":
    st.header("👤 Register Person to Database")
    st.markdown("Register a person using face recognition (ArcFace)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader("Upload Person Video", type=['mp4', 'avi', 'mkv', 'mov'])
        st.caption("Video should show clear face for best results")
    
    with col2:
        st.markdown("### Parameters")
        person_name = st.text_input("Person Name", placeholder="e.g., John Doe")
        global_id = st.number_input("Global ID", min_value=1, value=1, help="Unique ID for this person")
        sample_rate = st.number_input("Sample Rate", min_value=1, value=5, help="Extract 1 frame every N frames")
        delete_existing = st.checkbox("Delete Existing Collection", value=False, 
                                     help="⚠️ This will delete all registered persons!")
    
    if st.button("✅ Register Person", type="primary"):
        if video_file is None:
            st.error("Please upload a video file")
        elif not person_name:
            st.error("Please enter person name")
        else:
            with st.spinner(f"Uploading video and registering {person_name}..."):
                try:
                    # Prepare files and data for API call
                    files = {"video": (video_file.name, video_file.getvalue(), "video/mp4")}
                    data = {
                        "person_name": person_name,
                        "global_id": global_id,
                        "sample_rate": sample_rate,
                        "delete_existing": delete_existing
                    }

                    # Call Register API
                    response = requests.post(f"{REGISTER_API_URL}/register", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]

                        st.info(f"Job ID: {job_id}")

                        # Poll for status
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while True:
                            status_response = requests.get(f"{REGISTER_API_URL}/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()
                                status_text.text(f"Status: {status['status']}")

                                if status["status"] == "completed":
                                    progress_bar.progress(100)
                                    st.success(f"✅ {person_name} registered successfully!")
                                    st.info(f"Global ID: {global_id}")
                                    st.balloons()
                                    break

                                elif status["status"] == "failed":
                                    st.error(f"❌ Registration failed: {status.get('error', 'Unknown error')}")
                                    break

                                elif status["status"] == "processing":
                                    progress_bar.progress(50)

                            time.sleep(2)
                    else:
                        st.error(f"Failed to start registration: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 3: DETECT & TRACK
# ============================================================================
elif page == "🔍 Detect & Track":
    st.header("🔍 Detect & Track Persons")
    st.markdown("Detect and identify registered persons in video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader("Upload Video to Analyze", type=['mp4', 'avi', 'mkv', 'mov'])
    
    with col2:
        st.markdown("### Info")
        st.info("This will detect and track all registered persons in the video")

    if st.button("🚀 Start Detection", type="primary"):
        if video_file is None:
            st.error("Please upload a video file")
        else:
            with st.spinner("Uploading video and starting detection..."):
                try:
                    # Prepare files for API call
                    files = {"video": (video_file.name, video_file.getvalue(), "video/mp4")}

                    # Call Detection API
                    response = requests.post(f"{DETECTION_API_URL}/detect", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]

                        st.info(f"Job ID: {job_id}")

                        # Poll for status
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while True:
                            status_response = requests.get(f"{DETECTION_API_URL}/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()
                                status_text.text(f"Status: {status['status']}")

                                if status["status"] == "completed":
                                    progress_bar.progress(100)
                                    st.success("✅ Detection complete!")

                                    # Fetch results ONCE and cache in session state
                                    video_url = f"{DETECTION_API_URL}/download/video/{job_id}"
                                    csv_url = f"{DETECTION_API_URL}/download/csv/{job_id}"
                                    log_url = f"{DETECTION_API_URL}/download/log/{job_id}"

                                    # Cache video data
                                    video_cache_key = f"detect_video_{job_id}"
                                    if video_cache_key not in st.session_state:
                                        try:
                                            video_response = requests.get(video_url)
                                            if video_response.status_code == 200:
                                                st.session_state[video_cache_key] = video_response.content
                                        except Exception as e:
                                            st.error(f"Failed to fetch video: {e}")

                                    # Cache CSV data
                                    csv_cache_key = f"detect_csv_{job_id}"
                                    if csv_cache_key not in st.session_state:
                                        try:
                                            csv_response = requests.get(csv_url)
                                            if csv_response.status_code == 200:
                                                st.session_state[csv_cache_key] = csv_response.content
                                        except Exception as e:
                                            st.error(f"Failed to fetch CSV: {e}")

                                    # Cache log data
                                    log_cache_key = f"detect_log_{job_id}"
                                    if log_cache_key not in st.session_state:
                                        try:
                                            log_response = requests.get(log_url)
                                            if log_response.status_code == 200:
                                                st.session_state[log_cache_key] = log_response.content
                                        except Exception as e:
                                            st.error(f"Failed to fetch log: {e}")

                                    # Get cached data
                                    video_data = st.session_state.get(video_cache_key)
                                    csv_data = st.session_state.get(csv_cache_key)
                                    log_data = st.session_state.get(log_cache_key)

                                    # Video Preview
                                    if video_data:
                                        st.markdown("### 🎬 Video Preview")
                                        try:
                                            # Encode video to base64 for inline display (works with proxy)
                                            import base64

                                            video_base64 = base64.b64encode(video_data).decode()

                                            # Embed video using HTML5 video tag with base64 data URI
                                            video_html = f"""
                                            <video width="100%" controls>
                                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                                Your browser does not support the video tag.
                                            </video>
                                            """
                                            st.markdown(video_html, unsafe_allow_html=True)
                                            st.info(f"📹 Video size: {len(video_data) / (1024*1024):.2f} MB")
                                        except Exception as e:
                                            st.error(f"Failed to display video: {e}")
                                            st.info(f"Video size: {len(video_data)} bytes")

                                    # CSV Preview
                                    if csv_data:
                                        st.markdown("### 📊 Tracking Data Preview")
                                        import pandas as pd
                                        import io
                                        try:
                                            df = pd.read_csv(io.BytesIO(csv_data))
                                            st.dataframe(df.head(100), width="stretch")
                                            st.info(f"Showing first 100 rows of {len(df)} total detections")
                                        except Exception as e:
                                            st.error(f"Failed to parse CSV: {e}")

                                    # Download buttons - USE CACHED DATA
                                    st.markdown("### 📁 Download Results")
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        if video_data:
                                            st.download_button(
                                                label="📹 Download Video",
                                                data=video_data,
                                                file_name=f"{job_id}_output.mp4",
                                                mime="video/mp4",
                                                key=f"download_detect_video_{job_id}"
                                            )
                                        else:
                                            st.warning("Video not available")

                                    with col2:
                                        if csv_data:
                                            st.download_button(
                                                label="📊 Download CSV",
                                                data=csv_data,
                                                file_name=f"{job_id}_tracking.csv",
                                                mime="text/csv",
                                                key=f"download_detect_csv_{job_id}"
                                            )
                                        else:
                                            st.warning("CSV not available")

                                    with col3:
                                        if log_data:
                                            st.download_button(
                                                label="📄 Download Log",
                                                data=log_data,
                                                file_name=f"{job_id}_detection.log",
                                                mime="text/plain",
                                                key=f"download_detect_log_{job_id}"
                                            )
                                        else:
                                            st.warning("Log not available")

                                    break

                                elif status["status"] == "failed":
                                    st.error(f"❌ Detection failed: {status.get('error', 'Unknown error')}")
                                    break

                                elif status["status"] == "processing":
                                    progress_bar.progress(50)

                            time.sleep(2)
                    else:
                        st.error(f"Failed to start detection: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.header("ℹ️ About Person ReID System")
    
    st.markdown("""
    ### 🎯 System Overview
    
    Multi-camera person re-identification system using:
    - **YOLOX Detection**: High-accuracy person detection
    - **ByteTrack**: Multi-object tracking
    - **ArcFace**: Face recognition (512-dim embeddings)
    - **Qdrant**: Vector database for person storage
    
    ### 📊 Performance
    - **Speed**: ~19 FPS (optimized strategy)
    - **Accuracy**: 0.85-0.95 similarity for good matches
    - **Strategy**: First-3 + Re-verify every 30 frames
    
    ### 🔄 Workflow
    
    1. **Extract Objects** (Optional)
       - For multi-person videos
       - Creates separate video per person
    
    2. **Register Person**
       - Upload person video with clear face
       - System extracts face embeddings
       - Stores in Qdrant database
    
    3. **Detect & Track**
       - Upload video to analyze
       - System identifies registered persons
       - Outputs annotated video + CSV data
    
    ### 📁 Output Structure
    ```
    outputs/
    ├── videos/          # Annotated videos
    ├── csv/             # Tracking data
    ├── logs/            # Detailed logs
    └── extracted_objects/  # Extracted person videos
    ```
    
    ### 🎨 Features
    - Real-time FPS counter
    - Color-coded bounding boxes (Green=Known, Red=Unknown)
    - Similarity scores on labels
    - Detailed CSV tracking data
    - Per-frame event logs
    
    ### ⚙️ Configuration
    - Config file: `configs/config.yaml`
    - Qdrant credentials: `configs/.env`
    - Models: `models/` directory
    
    ---
    **Version**: 1.0  
    **Author**: Person ReID Team
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Tips")
st.sidebar.info("""
**Thresholds:**
- 0.8 = Strict (high precision)
- 0.7 = Balanced
- 0.6 = Loose (high recall)

**Best Practices:**
- Use clear face videos for registration
- Unique Global ID per person
- Test with max_frames first
""")

