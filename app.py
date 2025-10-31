#!/usr/bin/env python3
"""
Streamlit UI for Person Re-Identification System
Simple and clean interface for all operations - API Client Version
"""

import streamlit as st
import os
import requests
import time
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API endpoints from environment variables
EXTRACT_API_URL = os.getenv("EXTRACT_API_URL", "http://localhost:8001")
REGISTER_API_URL = os.getenv("REGISTER_API_URL", "http://localhost:8002")
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8003")

logger.info(f"🚀 Starting Person ReID UI - Extract: {EXTRACT_API_URL}, Register: {REGISTER_API_URL}, Detection: {DETECTION_API_URL}")

# Page config
st.set_page_config(
    page_title="Person ReID System",
    page_icon="",
    layout="wide"
)

# Title
st.title("Person Re-Identification System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Operation",
    ["Extract Objects", "Register Person", "Detect & Track", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: EXTRACT OBJECTS
# ============================================================================
if page == "Extract Objects":
    st.header("Extract Individual Objects from Video")
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
                    key=f"download_zip_{job_id}",
                    use_container_width=True
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
                    key=f"download_log_{job_id}",
                    use_container_width=True
                )

            # Button to clear results and browse new video
            st.markdown("---")
            if st.button("🔄 Clear Results & Browse New Video", type="secondary", use_container_width=True, key=f"clear_extract_{job_id}"):
                # Clear all extraction-related session state
                if 'extract_current_job_id' in st.session_state:
                    job_id_to_clear = st.session_state['extract_current_job_id']
                    st.session_state.pop('extract_current_job_id', None)
                    st.session_state.pop(f"extract_results_{job_id_to_clear}", None)
                    st.session_state.pop(f"extract_zip_{job_id_to_clear}", None)
                    st.session_state.pop(f"extract_log_{job_id_to_clear}", None)
                    # Clear individual file caches
                    for key in list(st.session_state.keys()):
                        if key.startswith(f"extract_file_{job_id_to_clear}"):
                            st.session_state.pop(key, None)
                st.rerun()

# ============================================================================
# PAGE 2: REGISTER PERSON
# ============================================================================
elif page == "Register Person":
    st.header("Register Person to Database")
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
elif page == "Detect & Track":
    st.header("Detect & Track Persons")
    st.markdown("Detect and identify registered persons in video")

    col1, col2 = st.columns([2, 1])

    with col1:
        video_file = st.file_uploader("Upload Video to Analyze", type=['mp4', 'avi', 'mkv', 'mov'])

    with col2:
        st.markdown("### Info")
        st.info("This will detect and track all registered persons in the video")

    # Zone Monitoring Section
    with st.expander("🗺️ Zone Monitoring (Optional)", expanded=False):
        st.markdown("### Working Zone Configuration")

        # Option to upload or create zones
        zone_input_method = st.radio(
            "Zone Configuration Method",
            ["Create Zones in UI", "Upload YAML File"],
            horizontal=True
        )

        zone_config_file = None
        zones_data = None

        if zone_input_method == "Upload YAML File":
            zone_config_file = st.file_uploader(
                "Upload Zone Config (YAML)",
                type=['yaml', 'yml'],
                help="YAML file defining zones and authorized persons"
            )

            if zone_config_file:
                st.success(f"✅ Zone config loaded: {zone_config_file.name}")

        else:  # Create Zones in UI
            st.markdown("#### Define Zones")

            # Initialize session state for zones
            if 'zones_config' not in st.session_state:
                st.session_state.zones_config = []

            # Number of zones
            num_zones = st.number_input(
                "Number of Zones",
                min_value=0,
                max_value=10,
                value=len(st.session_state.zones_config) if st.session_state.zones_config else 0,
                step=1
            )

            # Adjust zones list
            while len(st.session_state.zones_config) < num_zones:
                st.session_state.zones_config.append({
                    'name': f'Zone {len(st.session_state.zones_config) + 1}',
                    'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
                    'authorized_ids': []
                })
            while len(st.session_state.zones_config) > num_zones:
                st.session_state.zones_config.pop()

            # Configure each zone
            if num_zones > 0:
                for i, zone in enumerate(st.session_state.zones_config):
                    with st.expander(f"📍 {zone['name']}", expanded=True):
                        col1, col2 = st.columns(2)

                        with col1:
                            zone['name'] = st.text_input(
                                "Zone Name",
                                value=zone['name'],
                                key=f"zone_name_{i}"
                            )

                            # Authorized IDs
                            auth_ids_str = st.text_input(
                                "Authorized IDs (comma-separated)",
                                value=','.join(map(str, zone['authorized_ids'])),
                                key=f"zone_auth_{i}",
                                help="Example: 1,2,3"
                            )

                            # Parse authorized IDs
                            if auth_ids_str.strip():
                                try:
                                    zone['authorized_ids'] = [int(x.strip()) for x in auth_ids_str.split(',') if x.strip()]
                                except:
                                    st.warning("Invalid ID format. Use comma-separated numbers.")
                                    zone['authorized_ids'] = []
                            else:
                                zone['authorized_ids'] = []

                        with col2:
                            st.markdown("**Polygon Coordinates (x,y)**")
                            st.markdown("*Format: x1,y1; x2,y2; x3,y3; x4,y4*")

                            # Convert polygon to string
                            polygon_str = '; '.join([f"{p[0]},{p[1]}" for p in zone['polygon']])

                            polygon_input = st.text_area(
                                "Polygon Points",
                                value=polygon_str,
                                key=f"zone_polygon_{i}",
                                height=100,
                                help="Enter coordinates as: x1,y1; x2,y2; x3,y3; ..."
                            )

                            # Parse polygon
                            try:
                                points = []
                                for point_str in polygon_input.split(';'):
                                    point_str = point_str.strip()
                                    if point_str:
                                        x, y = map(float, point_str.split(','))
                                        points.append([int(x), int(y)])

                                if len(points) >= 3:
                                    zone['polygon'] = points
                                else:
                                    st.warning("Need at least 3 points for a polygon")
                            except:
                                st.warning("Invalid polygon format. Use: x1,y1; x2,y2; ...")

                        # Show zone info
                        st.info(f"✅ {len(zone['polygon'])} points, Authorized: {zone['authorized_ids']}")

                # Create YAML content from zones
                zones_dict = {}
                for i, zone in enumerate(st.session_state.zones_config):
                    zone_id = f"zone{i+1}"
                    zones_dict[zone_id] = {
                        'name': zone['name'],
                        'polygon': zone['polygon'],
                        'authorized_ids': zone['authorized_ids']
                    }

                zones_data = {'zones': zones_dict}

                # Preview YAML
                with st.expander("📄 Preview YAML Config", expanded=False):
                    yaml_content = yaml.dump(zones_data, default_flow_style=False, sort_keys=False)
                    st.code(yaml_content, language='yaml')

                    # Download button for YAML
                    st.download_button(
                        label="💾 Download Zone Config",
                        data=yaml_content,
                        file_name="zones.yaml",
                        mime="application/x-yaml"
                    )

        # IOU Threshold (common for both methods)
        st.markdown("---")
        iou_threshold = st.slider(
            "Zone IOU Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Minimum overlap (IOU) to consider person in zone (60% recommended)"
        )

    # Advanced Parameters
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        st.markdown("### Detection & Tracking Parameters")

        col_param1, col_param2 = st.columns(2)

        with col_param1:
            model_type = st.selectbox(
                "Model Type",
                options=["mot17", "yolox"],
                index=0,
                help="Detection model: mot17 (recommended) or yolox"
            )

            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Cosine similarity threshold for ReID matching (higher = stricter)"
            )

            conf_thresh = st.slider(
                "Detection Confidence",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Detection confidence threshold (higher = fewer detections)"
            )

        with col_param2:
            track_thresh = st.slider(
                "Tracking Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Tracking confidence threshold (higher = stricter tracking)"
            )

            st.markdown("**Current Settings:**")
            st.code(f"""
Model: {model_type}
Similarity: {similarity_threshold}
Detection: {conf_thresh}
Tracking: {track_thresh}
Zone Monitoring: {'Enabled' if zone_config_file else 'Disabled'}
IOU Threshold: {iou_threshold}
            """)

    if st.button("🚀 Start Detection", type="primary"):
        logger.info(f"📌 [Detect & Track] Start Detection button clicked")
        if video_file is None:
            logger.warning(f"⚠️ [Detect & Track] No video file uploaded")
            st.error("Please upload a video file")
        else:
            logger.info(f"📹 [Detect & Track] Uploading video: {video_file.name} ({len(video_file.getvalue()) / (1024*1024):.2f} MB)")
            logger.info(f"⚙️ [Detect & Track] Parameters: model={model_type}, similarity={similarity_threshold}, conf={conf_thresh}, track={track_thresh}")

            # Check if zone monitoring is enabled
            zone_enabled = zone_config_file is not None or zones_data is not None
            if zone_enabled:
                if zone_config_file:
                    logger.info(f"🗺️ [Detect & Track] Zone monitoring enabled (uploaded): {zone_config_file.name}")
                else:
                    logger.info(f"🗺️ [Detect & Track] Zone monitoring enabled (UI): {len(zones_data['zones'])} zones")

            with st.spinner("Uploading video and starting detection..."):
                try:
                    # Prepare files and data for API call
                    files = {"video": (video_file.name, video_file.getvalue(), "video/mp4")}

                    # Add zone config if provided
                    if zone_config_file:
                        # Use uploaded file
                        files["zone_config"] = (zone_config_file.name, zone_config_file.getvalue(), "application/x-yaml")
                    elif zones_data:
                        # Create YAML from UI data
                        yaml_content = yaml.dump(zones_data, default_flow_style=False, sort_keys=False)
                        files["zone_config"] = ("zones.yaml", yaml_content.encode('utf-8'), "application/x-yaml")

                    data = {
                        "similarity_threshold": similarity_threshold,
                        "model_type": model_type,
                        "conf_thresh": conf_thresh,
                        "track_thresh": track_thresh,
                        "iou_threshold": iou_threshold
                    }

                    # Call Detection API
                    logger.info(f"🔄 [Detect & Track] Calling detection API: {DETECTION_API_URL}/detect")
                    response = requests.post(f"{DETECTION_API_URL}/detect", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]
                        logger.info(f"✅ [Detect & Track] Detection job created: {job_id}")

                        # Store job_id in session state for display after rerun
                        st.session_state['detect_current_job_id'] = job_id

                        st.info(f"Job ID: {job_id}")

                        # Create placeholders for real-time updates
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        progress_text = st.empty()
                        tracks_container = st.empty()

                        poll_count = 0
                        while True:
                            poll_count += 1
                            # Get progress
                            try:
                                progress_response = requests.get(f"{DETECTION_API_URL}/progress/{job_id}")
                                if progress_response.status_code == 200:
                                    progress = progress_response.json()

                                    # Update progress bar
                                    progress_bar.progress(min(progress['progress_percent'] / 100, 0.99))

                                    # Update status text
                                    status_text.text(f"Status: {progress['status']}")

                                    # Update progress text
                                    progress_text.text(f"📊 Frame {progress['current_frame']}/{progress['total_frames']} ({progress['progress_percent']:.1f}%)")

                                    if poll_count % 10 == 0:  # Log every 10 polls
                                        logger.info(f"📊 [Detect & Track] Progress: {progress['current_frame']}/{progress['total_frames']} ({progress['progress_percent']:.1f}%)")

                                    # Display current tracks
                                    if progress['tracks']:
                                        tracks_info = "### 🎯 Current Tracks:\n"
                                        for track in progress['tracks']:
                                            color = "🟢" if track['label'] != "Unknown" else "🔴"
                                            tracks_info += f"{color} Track {track['track_id']}: **{track['label']}** (sim: {track['similarity']:.3f})\n"
                                        tracks_container.markdown(tracks_info)
                            except Exception as e:
                                logger.warning(f"⚠️ [Detect & Track] Progress fetch error: {e}")
                                pass

                            # Get status
                            status_response = requests.get(f"{DETECTION_API_URL}/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()

                                if status["status"] == "completed":
                                    progress_bar.progress(1.0)
                                    logger.info(f"✅ [Detect & Track] Detection completed: {job_id}")
                                    st.success("✅ Detection complete!")

                                    # Fetch results ONCE and cache in session state
                                    video_url = f"{DETECTION_API_URL}/download/video/{job_id}"
                                    csv_url = f"{DETECTION_API_URL}/download/csv/{job_id}"
                                    log_url = f"{DETECTION_API_URL}/download/log/{job_id}"

                                    # Cache video data
                                    video_cache_key = f"detect_video_{job_id}"
                                    if video_cache_key not in st.session_state:
                                        try:
                                            logger.info(f"📥 [Detect & Track] Fetching video: {video_url}")
                                            video_response = requests.get(video_url)
                                            if video_response.status_code == 200:
                                                st.session_state[video_cache_key] = video_response.content
                                                logger.info(f"✅ [Detect & Track] Video cached: {len(video_response.content) / (1024*1024):.2f} MB")
                                        except Exception as e:
                                            logger.error(f"❌ [Detect & Track] Failed to fetch video: {e}")
                                            st.error(f"Failed to fetch video: {e}")

                                    # Cache CSV data
                                    csv_cache_key = f"detect_csv_{job_id}"
                                    if csv_cache_key not in st.session_state:
                                        try:
                                            logger.info(f"📥 [Detect & Track] Fetching CSV: {csv_url}")
                                            csv_response = requests.get(csv_url)
                                            if csv_response.status_code == 200:
                                                st.session_state[csv_cache_key] = csv_response.content
                                                logger.info(f"✅ [Detect & Track] CSV cached: {len(csv_response.content) / 1024:.2f} KB")
                                        except Exception as e:
                                            logger.error(f"❌ [Detect & Track] Failed to fetch CSV: {e}")
                                            st.error(f"Failed to fetch CSV: {e}")

                                    # Cache log data
                                    log_cache_key = f"detect_log_{job_id}"
                                    if log_cache_key not in st.session_state:
                                        try:
                                            logger.info(f"📥 [Detect & Track] Fetching log: {log_url}")
                                            log_response = requests.get(log_url)
                                            if log_response.status_code == 200:
                                                st.session_state[log_cache_key] = log_response.content
                                                logger.info(f"✅ [Detect & Track] Log cached: {len(log_response.content) / 1024:.2f} KB")
                                        except Exception as e:
                                            logger.error(f"❌ [Detect & Track] Failed to fetch log: {e}")
                                            st.error(f"Failed to fetch log: {e}")

                                    # Cache zone JSON if zone monitoring was enabled
                                    if status.get("zone_monitoring", False):
                                        json_url = f"{DETECTION_API_URL}/download/json/{job_id}"
                                        json_cache_key = f"detect_json_{job_id}"
                                        if json_cache_key not in st.session_state:
                                            try:
                                                logger.info(f"📥 [Detect & Track] Fetching zone report: {json_url}")
                                                json_response = requests.get(json_url)
                                                if json_response.status_code == 200:
                                                    st.session_state[json_cache_key] = json_response.content
                                                    logger.info(f"✅ [Detect & Track] Zone report cached: {len(json_response.content) / 1024:.2f} KB")
                                            except Exception as e:
                                                logger.warning(f"⚠️ [Detect & Track] Failed to fetch zone report: {e}")

                                    break

                                elif status["status"] == "failed":
                                    logger.error(f"❌ [Detect & Track] Detection failed: {status.get('error', 'Unknown error')}")
                                    st.error(f"❌ Detection failed: {status.get('error', 'Unknown error')}")
                                    break

                            time.sleep(1)
                    else:
                        logger.error(f"❌ [Detect & Track] Failed to start detection: {response.text}")
                        st.error(f"Failed to start detection: {response.text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display results if available (after rerun from download button click)
    if 'detect_current_job_id' in st.session_state:
        job_id = st.session_state['detect_current_job_id']
        logger.info(f"📋 [Detect & Track] Displaying results for job: {job_id}")

        # Get cached data
        video_cache_key = f"detect_video_{job_id}"
        csv_cache_key = f"detect_csv_{job_id}"
        log_cache_key = f"detect_log_{job_id}"
        json_cache_key = f"detect_json_{job_id}"

        video_data = st.session_state.get(video_cache_key)
        csv_data = st.session_state.get(csv_cache_key)
        log_data = st.session_state.get(log_cache_key)
        json_data = st.session_state.get(json_cache_key)

        logger.info(f"📊 [Detect & Track] Cache status - Video: {bool(video_data)}, CSV: {bool(csv_data)}, Log: {bool(log_data)}, JSON: {bool(json_data)}")

        # Zone Report Preview (if available)
        if json_data:
            st.markdown("### 🗺️ Zone Monitoring Report")
            import json
            import io
            try:
                logger.info(f"🗺️ [Detect & Track] Parsing zone report...")
                zone_report = json.loads(json_data)

                # Display summary
                if "summary" in zone_report:
                    st.markdown("#### Zone Summary")
                    for zone_id, zone_info in zone_report["summary"].items():
                        with st.expander(f"📍 {zone_info['name']} ({zone_id})", expanded=True):
                            st.markdown(f"**Authorized IDs:** {zone_info['authorized_ids']}")
                            st.markdown(f"**Current Persons:** {zone_info['count']}")

                            if zone_info['current_persons']:
                                for person in zone_info['current_persons']:
                                    status_icon = "✅" if person['authorized'] else "⚠️"
                                    st.markdown(f"{status_icon} **{person['name']}** (ID: {person['id']}) - {person['duration']:.1f}s")

                # Display violations
                if "violations" in zone_report and zone_report["violations"]:
                    st.markdown("#### ⚠️ Violations Detected")
                    for violation in zone_report["violations"]:
                        st.warning(f"🚫 **{violation['name']}** (ID: {violation['global_id']}) entered unauthorized zone **{violation['zone_name']}** at {violation['time']:.1f}s")

                logger.info(f"✅ [Detect & Track] Zone report displayed")
            except Exception as e:
                logger.error(f"❌ [Detect & Track] Failed to parse zone report: {e}")
                st.error(f"Failed to parse zone report: {e}")

        # CSV Preview
        if csv_data:
            st.markdown("### 📊 Tracking Data Preview")
            import pandas as pd
            import io
            try:
                logger.info(f"📊 [Detect & Track] Parsing CSV...")
                df = pd.read_csv(io.BytesIO(csv_data))
                st.dataframe(df.head(100), width="stretch")
                st.info(f"Showing first 100 rows of {len(df)} total detections")
                logger.info(f"✅ [Detect & Track] CSV displayed: {len(df)} rows")
            except Exception as e:
                logger.error(f"❌ [Detect & Track] Failed to parse CSV: {e}")
                st.error(f"Failed to parse CSV: {e}")

        # Download buttons - USE CACHED DATA
        st.markdown("### 📁 Download Results")

        # Adjust columns based on whether zone report exists
        if json_data:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)
            col4 = None

        with col1:
            if video_data:
                if st.download_button(
                    label="📹 Download Video",
                    data=video_data,
                    file_name=f"{job_id}_output.mp4",
                    mime="video/mp4",
                    key=f"download_detect_video_{job_id}",
                    use_container_width=True
                ):
                    logger.info(f"📥 [Detect & Track] User downloading video: {job_id}_output.mp4 ({len(video_data) / (1024*1024):.2f} MB)")
            else:
                st.warning("Video not available")

        with col2:
            if csv_data:
                if st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"{job_id}_tracking.csv",
                    mime="text/csv",
                    key=f"download_detect_csv_{job_id}",
                    use_container_width=True
                ):
                    logger.info(f"📥 [Detect & Track] User downloading CSV: {job_id}_tracking.csv ({len(csv_data) / 1024:.2f} KB)")
            else:
                st.warning("CSV not available")

        with col3:
            if log_data:
                if st.download_button(
                    label="📄 Download Log",
                    data=log_data,
                    file_name=f"{job_id}_detection.log",
                    mime="text/plain",
                    key=f"download_detect_log_{job_id}",
                    use_container_width=True
                ):
                    logger.info(f"📥 [Detect & Track] User downloading log: {job_id}_detection.log ({len(log_data) / 1024:.2f} KB)")
            else:
                st.warning("Log not available")

        if col4 and json_data:
            if st.download_button(
                label="🗺️ Download Zone Report",
                data=json_data,
                file_name=f"{job_id}_zones.json",
                mime="application/json",
                key=f"download_detect_json_{job_id}",
                use_container_width=True
            ):
                logger.info(f"📥 [Detect & Track] User downloading zone report: {job_id}_zones.json ({len(json_data) / 1024:.2f} KB)")

        # Button to clear results and browse new video
        st.markdown("---")
        if st.button("🔄 Clear Results & Browse New Video", type="secondary", use_container_width=True):
            logger.info(f"🔄 [Detect & Track] User clearing results for job: {job_id}")
            # Clear all detection-related session state
            if 'detect_current_job_id' in st.session_state:
                job_id_to_clear = st.session_state['detect_current_job_id']
                st.session_state.pop('detect_current_job_id', None)
                st.session_state.pop(f"detect_video_{job_id_to_clear}", None)
                st.session_state.pop(f"detect_csv_{job_id_to_clear}", None)
                st.session_state.pop(f"detect_log_{job_id_to_clear}", None)
                logger.info(f"✅ [Detect & Track] Results cleared, ready for new video")
            st.rerun()

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

