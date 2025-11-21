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
import asyncio
import websockets
import json
import threading
from queue import Queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API endpoints from environment variables
REGISTER_API_URL = os.getenv("REGISTER_API_URL", "http://localhost:8002")
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8003")

logger.info(f"🚀 Starting Person ReID UI - Register: {REGISTER_API_URL}, Detection: {DETECTION_API_URL}")


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

# Import database manager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from services.database import PostgresManager

# Initialize database manager (singleton pattern)
@st.cache_resource
def get_db_manager():
    """Get or create database manager instance"""
    try:
        db_manager = PostgresManager()
        db_manager.connect()
        logger.info("✅ Connected to PostgreSQL database")
        return db_manager
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_users_from_db():
    """
    Fetch all users from PostgreSQL database
    Returns list of users or empty list on error
    """
    try:
        db_manager = get_db_manager()
        if db_manager:
            users = db_manager.get_all_users()
            return [user.dict() for user in users]
        return []
    except Exception as e:
        logger.error(f"Error fetching users from database: {e}")
        return []


def fetch_users_dict():
    """
    Fetch users as dictionary mapping global_id to name
    Returns dict or empty dict on error
    """
    try:
        db_manager = get_db_manager()
        if db_manager:
            return db_manager.get_users_dict()
        return {}
    except Exception as e:
        logger.error(f"Error fetching users dict from database: {e}")
        return {}


# ============================================================================
# WEBSOCKET CLIENT FOR REALTIME VIOLATION LOGS
# ============================================================================

def websocket_client_thread(job_id: str, message_queue: Queue, detection_api_url: str):
    """
    WebSocket client thread to receive realtime violation logs
    Runs in background and puts messages in queue for UI to display
    """
    async def connect_and_listen():
        ws_url = f"ws://{detection_api_url.replace('http://', '')}/ws/violations/{job_id}"
        logger.info(f"🔌 WebSocket connecting to: {ws_url}")
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as websocket:
                logger.info(f"✅ WebSocket connected to {ws_url}")
                message_queue.put({"type": "connection", "status": "connected"})

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        logger.debug(f"📨 WebSocket message received: {data.get('zone', 'Unknown')}")
                        message_queue.put({"type": "violation", "data": data})
                    except asyncio.TimeoutError:
                        # Keep-alive timeout, continue
                        continue
                    except Exception as e:
                        logger.debug(f"WebSocket receive error: {e}")
                        break
        except Exception as e:
            logger.warning(f"⚠️ WebSocket connection failed: {e}")
            message_queue.put({"type": "connection", "status": "failed", "error": str(e)})

    # Run async event loop in thread
    try:
        asyncio.run(connect_and_listen())
    except Exception as e:
        logger.error(f"WebSocket thread error: {e}")
        message_queue.put({"type": "connection", "status": "error", "error": str(e)})

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
    ["Detect & Track", "Register Person", "🗄️ DB Management", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: REGISTER PERSON
# ============================================================================
if page == "Register Person":
    st.header("Register Person to Database")
    st.markdown("Register a person using face recognition (ArcFace)")

    # Input type selection
    input_type = st.radio("Input Type", ["📹 Video", "🖼️ Images"], horizontal=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        if input_type == "📹 Video":
            # Toggle between single and batch upload
            upload_mode = st.radio("Upload Mode", ["Single Video", "Multiple Videos"], horizontal=True)

            if upload_mode == "Single Video":
                video_files = st.file_uploader("Upload Person Video", type=['mp4', 'avi', 'mkv', 'mov'])
                if video_files:
                    video_files = [video_files]
            else:
                video_files = st.file_uploader("Upload Person Videos", type=['mp4', 'avi', 'mkv', 'mov'], accept_multiple_files=True)

            st.caption("Video(s) should show clear face for best results")
            image_files = None
        else:
            # Image upload
            image_files = st.file_uploader("Upload Person Images", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
            st.caption("Upload multiple images showing clear face from different angles")
            video_files = None

    with col2:
        st.markdown("### Parameters")

        # Fetch users from database
        users_dict = fetch_users_dict()

        if users_dict:
            # Create options for selectbox: "Name (ID: global_id)"
            user_options = {f"{name} (ID: {gid})": (gid, name) for gid, name in users_dict.items()}
            user_options_list = ["➕ Create New User"] + list(user_options.keys())

            selected_option = st.selectbox(
                "Select User",
                options=user_options_list,
                help="Select existing user or create new one"
            )

            if selected_option == "➕ Create New User":
                # Manual input for new user
                person_name = st.text_input("Person Name", placeholder="e.g., John Doe")
                global_id = st.number_input("Global ID", min_value=1, value=1, help="Unique ID for this person")
            else:
                # Use selected user
                global_id, person_name = user_options[selected_option]
                st.info(f"✅ Selected: **{person_name}** (Global ID: **{global_id}**)")
        else:
            # Fallback to manual input if database is not available
            st.warning("⚠️ Database not available. Using manual input.")
            person_name = st.text_input("Person Name", placeholder="e.g., John Doe")
            global_id = st.number_input("Global ID", min_value=1, value=1, help="Unique ID for this person")

        if input_type == "📹 Video":
            sample_rate = st.number_input("Sample Rate", min_value=1, value=5, help="Extract 1 frame every N frames")
        else:
            sample_rate = None

        face_conf_thresh = st.slider(
            "Face Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Face detection confidence threshold (higher = stricter face detection)"
        )

        delete_existing = st.checkbox("Delete Existing Collection", value=False,
                                     help="⚠️ This will delete all registered persons!")

    if st.button("✅ Register Person", type="primary"):
        # Validation
        if input_type == "📹 Video":
            if not video_files:
                st.error("Please upload at least one video file")
            elif not person_name:
                st.error("Please enter person name")
            else:
                # Video registration
                with st.spinner(f"Uploading {len(video_files)} video(s) and registering {person_name}..."):
                    try:
                        # Prepare files and data for API call
                        files = [("videos", (vf.name, vf.getvalue(), "video/mp4")) for vf in video_files]
                        data = {
                            "person_name": person_name,
                            "global_id": global_id,
                            "sample_rate": sample_rate,
                            "face_conf_thresh": face_conf_thresh,
                            "delete_existing": delete_existing
                        }

                        # Call Register API (batch or single)
                        if len(video_files) == 1:
                            # Use single endpoint
                            files_dict = {"video": (video_files[0].name, video_files[0].getvalue(), "video/mp4")}
                            response = requests.post(f"{REGISTER_API_URL}/register", files=files_dict, data=data)
                            job_ids = [response.json()["job_id"]] if response.status_code == 200 else []
                        else:
                            # Use batch endpoint
                            response = requests.post(f"{REGISTER_API_URL}/register-batch", files=files, data=data)
                            job_ids = response.json()["job_ids"] if response.status_code == 200 else []

                        if response.status_code == 200:
                            st.info(f"Started {len(job_ids)} registration job(s)")

                            # Create progress tracking for all jobs
                            progress_bars = {}
                            status_texts = {}

                            for i, job_id in enumerate(job_ids):
                                with st.expander(f"Video {i+1} - {video_files[i].name}", expanded=True):
                                    progress_bars[job_id] = st.progress(0)
                                    status_texts[job_id] = st.empty()

                            # Poll for status of all jobs
                            all_completed = False
                            while not all_completed:
                                all_completed = True

                                for job_id in job_ids:
                                    status_response = requests.get(f"{REGISTER_API_URL}/status/{job_id}")
                                    if status_response.status_code == 200:
                                        status = status_response.json()
                                        status_texts[job_id].text(f"Status: {status['status']}")

                                        if status["status"] == "completed":
                                            progress_bars[job_id].progress(100)
                                        elif status["status"] == "failed":
                                            status_texts[job_id].error(f"❌ Failed: {status.get('error', 'Unknown error')}")
                                        elif status["status"] == "processing":
                                            progress_bars[job_id].progress(50)
                                            all_completed = False
                                        else:  # pending
                                            progress_bars[job_id].progress(25)
                                            all_completed = False

                                if not all_completed:
                                    time.sleep(2)

                            st.success(f"✅ {person_name} registered successfully with {len(job_ids)} video(s)!")
                            st.info(f"Global ID: {global_id}")
                            st.balloons()
                        else:
                            st.error(f"Failed to start registration: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        else:
            # Image registration
            if not image_files:
                st.error("Please upload at least one image file")
            elif not person_name:
                st.error("Please enter person name")
            else:
                with st.spinner(f"Uploading {len(image_files)} image(s) and registering {person_name}..."):
                    try:
                        # Prepare files and data for API call
                        files = [("images", (img.name, img.getvalue(), "image/jpeg")) for img in image_files]
                        data = {
                            "person_name": person_name,
                            "global_id": global_id,
                            "face_conf_thresh": face_conf_thresh,
                            "delete_existing": delete_existing
                        }

                        # Call Register Images API
                        response = requests.post(f"{REGISTER_API_URL}/register-images", files=files, data=data)

                        if response.status_code == 200:
                            job_id = response.json()["job_id"]
                            st.info(f"Started image registration job")

                            # Create progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Poll for status
                            completed = False
                            while not completed:
                                status_response = requests.get(f"{REGISTER_API_URL}/status/{job_id}")
                                if status_response.status_code == 200:
                                    status = status_response.json()
                                    status_text.text(f"Status: {status['status']}")

                                    if status["status"] == "completed":
                                        progress_bar.progress(100)
                                        completed = True
                                    elif status["status"] == "failed":
                                        status_text.error(f"❌ Failed: {status.get('error', 'Unknown error')}")
                                        completed = True
                                    elif status["status"] == "processing":
                                        progress_bar.progress(50)
                                    else:  # pending
                                        progress_bar.progress(25)

                                if not completed:
                                    time.sleep(2)

                            if status["status"] == "completed":
                                st.success(f"✅ {person_name} registered successfully with {len(image_files)} image(s)!")
                                st.info(f"Global ID: {global_id}")
                                st.balloons()
                        else:
                            st.error(f"Failed to start registration: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 3: DETECT & TRACK
# ============================================================================
elif page == "Detect & Track":
    st.header("Detect & Track Persons")
    st.markdown("Detect and identify registered persons in video or stream")

    # Input method selection
    input_method = st.radio(
        "Input Source",
        ["Upload Video File", "Stream URL (UDP/RTSP)"],
        horizontal=True
    )

    video_file = None
    stream_url = None
    max_frames = None
    max_duration = None

    if input_method == "Upload Video File":
        col1, col2 = st.columns([2, 1])

        with col1:
            video_file = st.file_uploader("Upload Video to Analyze", type=['mp4', 'avi', 'mkv', 'mov'])

        with col2:
            st.markdown("### Info")
            st.info("This will detect and track all registered persons in the video")

    else:  # Stream URL
        col1, col2 = st.columns([2, 1])

        with col1:
            stream_url = st.text_area(
                "Stream URL(s)",
                value="udp://127.0.0.1:1905",
                height=100,
                help="Enter one or more stream URLs:\n- Single camera: udp://127.0.0.1:1905\n- Multiple cameras (comma-separated): udp://127.0.0.1:1905, udp://127.0.0.1:1906\n- Multiple cameras (newline-separated): one URL per line"
            )

            st.markdown("### Stream Limits")
            limit_type = st.radio(
                "Limit by",
                ["Duration (seconds)", "Number of frames", "No limit"],
                horizontal=True
            )

            if limit_type == "Duration (seconds)":
                max_duration = st.number_input(
                    "Maximum duration (seconds)",
                    min_value=1,
                    max_value=3600,
                    value=60,
                    help="How many seconds of stream to process"
                )
            elif limit_type == "Number of frames":
                max_frames = st.number_input(
                    "Maximum frames",
                    min_value=1,
                    max_value=100000,
                    value=1800,
                    help="How many frames to process (e.g., 1800 frames = 60s at 30fps)"
                )

        with col2:
            st.markdown("### Info")
            st.info("⚠️ Stream will be processed in real-time. Output video will be saved with the specified limit.")
            if stream_url:
                st.code(f"Stream: {stream_url}", language="text")

    # Zone Monitoring Section
    with st.expander("🗺️ Zone Monitoring (Optional)", expanded=False):
        st.markdown("### Working Zone Configuration")

        # Option to upload or create zones
        zone_input_method = st.radio(
            "Zone Configuration Method",
            ["Create Zones in UI", "Upload Config File"],
            horizontal=True
        )

        zone_config_file = None
        zones_data = None

        if zone_input_method == "Upload Config File":
            zone_config_file = st.file_uploader(
                "Upload Zone Config (YAML/JSON)",
                type=['yaml', 'yml', 'json'],
                help="YAML or JSON file defining zones and authorized persons"
            )

            if zone_config_file:
                st.success(f"✅ Zone config loaded: {zone_config_file.name}")

        else:  # Create Zones in UI
            st.markdown("#### Define Zones")

            # Detect number of cameras from stream URL
            num_cameras = 1
            if input_method == "Stream URL (UDP/RTSP)" and stream_url:
                # Parse stream URLs
                urls = [u.strip() for u in stream_url.replace('\n', ',').split(',') if u.strip()]
                num_cameras = len(urls)

                # Debug info
                st.caption(f"🔍 Debug: Detected {num_cameras} camera(s) from stream URLs")

                if num_cameras > 1:
                    st.info(f"📹 Detected {num_cameras} cameras. You can configure zones per camera.")
                    st.caption(f"Stream URLs: {urls}")

            # Initialize session state for zones
            if 'zones_config' not in st.session_state:
                st.session_state.zones_config = []

            # Multi-camera mode: organize zones by camera
            if num_cameras > 1:
                st.markdown(f"**Configure zones for {num_cameras} cameras**")

                # Initialize camera zones structure
                if 'camera_zones' not in st.session_state:
                    st.session_state.camera_zones = {f'camera_{i+1}': [] for i in range(num_cameras)}

                # Ensure we have entries for all cameras
                for i in range(num_cameras):
                    camera_key = f'camera_{i+1}'
                    if camera_key not in st.session_state.camera_zones:
                        st.session_state.camera_zones[camera_key] = []

                # Configure zones for each camera
                for cam_idx in range(num_cameras):
                    camera_key = f'camera_{cam_idx+1}'

                    with st.expander(f"📹 Camera {cam_idx+1}", expanded=cam_idx==0):
                        num_zones_cam = st.number_input(
                            f"Number of Zones for Camera {cam_idx+1}",
                            min_value=0,
                            max_value=10,
                            value=len(st.session_state.camera_zones[camera_key]),
                            step=1,
                            key=f"num_zones_cam_{cam_idx}"
                        )

                        # Adjust zones list for this camera
                        while len(st.session_state.camera_zones[camera_key]) < num_zones_cam:
                            st.session_state.camera_zones[camera_key].append({
                                'name': f'Zone {len(st.session_state.camera_zones[camera_key]) + 1}',
                                'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
                                'authorized_ids': []
                            })
                        while len(st.session_state.camera_zones[camera_key]) > num_zones_cam:
                            st.session_state.camera_zones[camera_key].pop()

                        # Configure each zone for this camera
                        for zone_idx, zone in enumerate(st.session_state.camera_zones[camera_key]):
                            st.markdown(f"**Zone {zone_idx+1}**")
                            col1, col2 = st.columns(2)

                            with col1:
                                # Load zones from database for dropdown
                                db_zones = []
                                db_manager = None
                                try:
                                    db_manager = get_db_manager()
                                    if db_manager:
                                        db_zones = db_manager.get_all_zones()
                                except Exception as e:
                                    logger.warning(f"Could not load zones from database: {e}")

                                # Create zone name options
                                zone_name_options = ["Custom (Manual Input)"] + [f"{z.zone_name} ({z.zone_id})" for z in db_zones]

                                # Initialize session state for this zone's selection if not exists
                                selection_key = f"zone_selection_cam{cam_idx}_z{zone_idx}"
                                if selection_key not in st.session_state:
                                    st.session_state[selection_key] = "Custom (Manual Input)"

                                selected_zone_option = st.selectbox(
                                    "Zone Name",
                                    options=zone_name_options,
                                    index=zone_name_options.index(st.session_state[selection_key]) if st.session_state[selection_key] in zone_name_options else 0,
                                    key=f"zone_name_select_cam{cam_idx}_z{zone_idx}",
                                    help="Select from database or create custom zone"
                                )

                                # Check if selection changed
                                if selected_zone_option != st.session_state[selection_key]:
                                    st.session_state[selection_key] = selected_zone_option
                                    logger.info(f"Zone selection changed to: {selected_zone_option}")

                                    # If database zone selected, auto-fill immediately
                                    if selected_zone_option != "Custom (Manual Input)":
                                        zone_id = selected_zone_option.split('(')[-1].rstrip(')')
                                        logger.info(f"Auto-filling zone_id: {zone_id}")

                                        # Find the selected zone in database
                                        for z in db_zones:
                                            if z.zone_id == zone_id:
                                                # Auto-fill from database
                                                zone['name'] = z.zone_name
                                                zone['polygon'] = [
                                                    [int(z.x1), int(z.y1)],
                                                    [int(z.x2), int(z.y2)],
                                                    [int(z.x3), int(z.y3)],
                                                    [int(z.x4), int(z.y4)]
                                                ]
                                                logger.info(f"Auto-filled polygon: {zone['polygon']}")

                                                # Force update polygon widget state
                                                widget_key = f"zone_polygon_input_cam{cam_idx}_z{zone_idx}"
                                                polygon_str = '; '.join([f"{p[0]},{p[1]}" for p in zone['polygon']])
                                                st.session_state[widget_key] = polygon_str
                                                logger.info(f"Updated widget state {widget_key} = {polygon_str}")

                                                # Get authorized users from this zone
                                                if db_manager:
                                                    try:
                                                        users_in_zone = db_manager.get_users_by_zone(zone_id)
                                                        zone['authorized_ids'] = [u.global_id for u in users_in_zone]
                                                        logger.info(f"Auto-filled authorized_ids: {zone['authorized_ids']}")
                                                    except Exception as e:
                                                        logger.warning(f"Could not load users for zone: {e}")
                                                break
                                    st.rerun()

                                # Handle zone selection for display
                                if selected_zone_option == "Custom (Manual Input)":
                                    # Manual input
                                    zone['name'] = st.text_input(
                                        "Custom Zone Name",
                                        value=zone.get('name', f'Zone {zone_idx+1}'),
                                        key=f"zone_name_custom_cam{cam_idx}_z{zone_idx}"
                                    )
                                else:
                                    # Extract zone_id from selection
                                    zone_id = selected_zone_option.split('(')[-1].rstrip(')')

                                    # Find the selected zone in database and ensure data is loaded
                                    selected_db_zone = None
                                    for z in db_zones:
                                        if z.zone_id == zone_id:
                                            selected_db_zone = z
                                            break

                                    if selected_db_zone:
                                        # Ensure zone data is populated (in case rerun didn't happen)
                                        if not zone.get('polygon') or zone['polygon'] == [[100, 100], [200, 100], [200, 200], [100, 200]]:
                                            zone['name'] = selected_db_zone.zone_name
                                            zone['polygon'] = [
                                                [int(selected_db_zone.x1), int(selected_db_zone.y1)],
                                                [int(selected_db_zone.x2), int(selected_db_zone.y2)],
                                                [int(selected_db_zone.x3), int(selected_db_zone.y3)],
                                                [int(selected_db_zone.x4), int(selected_db_zone.y4)]
                                            ]

                                            # Get authorized users from this zone
                                            if db_manager:
                                                try:
                                                    users_in_zone = db_manager.get_users_by_zone(zone_id)
                                                    zone['authorized_ids'] = [u.global_id for u in users_in_zone]
                                                except Exception as e:
                                                    logger.warning(f"Could not load users for zone: {e}")

                                        st.info(f"✅ Zone loaded from database: {len(zone.get('authorized_ids', []))} authorized users")

                                # Fetch users from database for dropdown
                                users_dict = fetch_users_dict()

                                if users_dict:
                                    # Create options for multiselect: "Name (ID: global_id)"
                                    user_options = {f"{name} (ID: {gid})": gid for gid, name in users_dict.items()}

                                    # Get current selected options
                                    current_selections = []
                                    for auth_id in zone['authorized_ids']:
                                        if auth_id in users_dict:
                                            current_selections.append(f"{users_dict[auth_id]} (ID: {auth_id})")

                                    logger.info(f"[Multi-cam Zone {cam_idx+1}-{zone_idx+1}] current authorized_ids: {zone['authorized_ids']}, current_selections: {current_selections}")

                                    selected_users = st.multiselect(
                                        "Authorized Users",
                                        options=list(user_options.keys()),
                                        default=current_selections,
                                        key=f"zone_auth_cam{cam_idx}_z{zone_idx}",
                                        help="Select authorized users from database"
                                    )

                                    logger.info(f"[Multi-cam Zone {cam_idx+1}-{zone_idx+1}] selected_users from widget: {selected_users}")

                                    # Update authorized_ids based on selection
                                    # IMPORTANT: Only update if widget is not empty OR if current_selections was also empty
                                    # This prevents widget reset from clearing authorized_ids
                                    if selected_users or not current_selections:
                                        zone['authorized_ids'] = [user_options[user] for user in selected_users]
                                        logger.info(f"[Multi-cam Zone {cam_idx+1}-{zone_idx+1}] final authorized_ids: {zone['authorized_ids']}")
                                    else:
                                        logger.warning(f"[Multi-cam Zone {cam_idx+1}-{zone_idx+1}] Widget returned empty but current_selections was not empty - keeping existing authorized_ids: {zone['authorized_ids']}")
                                else:
                                    # Fallback to text input if database is not available
                                    st.warning("⚠️ Database not available. Using manual input.")
                                    auth_ids_str = st.text_input(
                                        "Authorized IDs",
                                        value=','.join(map(str, zone['authorized_ids'])),
                                        key=f"zone_auth_cam{cam_idx}_z{zone_idx}",
                                        help="Comma-separated: 1,2,3"
                                    )

                                    if auth_ids_str.strip():
                                        try:
                                            zone['authorized_ids'] = [int(x.strip()) for x in auth_ids_str.split(',') if x.strip()]
                                        except:
                                            zone['authorized_ids'] = []
                                    else:
                                        zone['authorized_ids'] = []

                            with col2:
                                # Convert current polygon to string
                                current_polygon_str = '; '.join([f"{p[0]},{p[1]}" for p in zone['polygon']])

                                # Use widget key directly - Streamlit will manage the state
                                widget_key = f"zone_polygon_input_cam{cam_idx}_z{zone_idx}"

                                # Initialize widget with current polygon value if not exists
                                if widget_key not in st.session_state:
                                    st.session_state[widget_key] = current_polygon_str

                                polygon_input = st.text_area(
                                    "Polygon (x,y; x,y; ...)",
                                    value=st.session_state[widget_key],
                                    key=widget_key,
                                    height=80
                                )

                                # Parse and update zone polygon from user input
                                try:
                                    points = []
                                    for point_str in polygon_input.split(';'):
                                        point_str = point_str.strip()
                                        if point_str:
                                            x, y = map(float, point_str.split(','))
                                            points.append([int(x), int(y)])
                                    if len(points) >= 3:
                                        zone['polygon'] = points
                                except:
                                    pass

                            st.caption(f"✅ {len(zone['polygon'])} points, Auth: {zone['authorized_ids']}")
                            st.divider()

            else:
                # Single camera mode (original logic)
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

                # Configure each zone (single camera)
                if num_zones > 0:
                    for i, zone in enumerate(st.session_state.zones_config):
                        with st.expander(f"📍 {zone['name']}", expanded=True):
                            col1, col2 = st.columns(2)

                            with col1:
                                # Load zones from database for dropdown
                                db_zones = []
                                db_manager = None
                                try:
                                    db_manager = get_db_manager()
                                    if db_manager:
                                        db_zones = db_manager.get_all_zones()
                                except Exception as e:
                                    logger.warning(f"Could not load zones from database: {e}")

                                # Create zone name options
                                zone_name_options = ["Custom (Manual Input)"] + [f"{z.zone_name} ({z.zone_id})" for z in db_zones]

                                # Initialize session state for this zone's selection if not exists
                                selection_key = f"zone_selection_{i}"
                                if selection_key not in st.session_state:
                                    st.session_state[selection_key] = "Custom (Manual Input)"

                                selected_zone_option = st.selectbox(
                                    "Zone Name",
                                    options=zone_name_options,
                                    index=zone_name_options.index(st.session_state[selection_key]) if st.session_state[selection_key] in zone_name_options else 0,
                                    key=f"zone_name_select_{i}",
                                    help="Select from database or create custom zone"
                                )

                                # Check if selection changed
                                if selected_zone_option != st.session_state[selection_key]:
                                    st.session_state[selection_key] = selected_zone_option
                                    logger.info(f"Zone selection changed to: {selected_zone_option}")

                                    # If database zone selected, auto-fill immediately
                                    if selected_zone_option != "Custom (Manual Input)":
                                        zone_id = selected_zone_option.split('(')[-1].rstrip(')')
                                        logger.info(f"Auto-filling zone_id: {zone_id}")

                                        # Find the selected zone in database
                                        for z in db_zones:
                                            if z.zone_id == zone_id:
                                                # Auto-fill from database
                                                zone['name'] = z.zone_name
                                                zone['polygon'] = [
                                                    [int(z.x1), int(z.y1)],
                                                    [int(z.x2), int(z.y2)],
                                                    [int(z.x3), int(z.y3)],
                                                    [int(z.x4), int(z.y4)]
                                                ]
                                                logger.info(f"Auto-filled polygon: {zone['polygon']}")

                                                # Force update polygon widget state
                                                widget_key = f"zone_polygon_input_{i}"
                                                polygon_str = '; '.join([f"{p[0]},{p[1]}" for p in zone['polygon']])
                                                st.session_state[widget_key] = polygon_str
                                                logger.info(f"Updated widget state {widget_key} = {polygon_str}")

                                                # Get authorized users from this zone
                                                if db_manager:
                                                    try:
                                                        users_in_zone = db_manager.get_users_by_zone(zone_id)
                                                        zone['authorized_ids'] = [u.global_id for u in users_in_zone]
                                                        logger.info(f"Auto-filled authorized_ids: {zone['authorized_ids']}")
                                                    except Exception as e:
                                                        logger.warning(f"Could not load users for zone: {e}")
                                                break
                                    st.rerun()

                                # Handle zone selection for display
                                if selected_zone_option == "Custom (Manual Input)":
                                    # Manual input
                                    zone['name'] = st.text_input(
                                        "Custom Zone Name",
                                        value=zone.get('name', f'Zone {i+1}'),
                                        key=f"zone_name_custom_{i}"
                                    )
                                else:
                                    # Extract zone_id from selection
                                    zone_id = selected_zone_option.split('(')[-1].rstrip(')')

                                    # Find the selected zone in database and ensure data is loaded
                                    selected_db_zone = None
                                    for z in db_zones:
                                        if z.zone_id == zone_id:
                                            selected_db_zone = z
                                            break

                                    if selected_db_zone:
                                        # Ensure zone data is populated (in case rerun didn't happen)
                                        if not zone.get('polygon') or zone['polygon'] == [[100, 100], [200, 100], [200, 200], [100, 200]]:
                                            zone['name'] = selected_db_zone.zone_name
                                            zone['polygon'] = [
                                                [int(selected_db_zone.x1), int(selected_db_zone.y1)],
                                                [int(selected_db_zone.x2), int(selected_db_zone.y2)],
                                                [int(selected_db_zone.x3), int(selected_db_zone.y3)],
                                                [int(selected_db_zone.x4), int(selected_db_zone.y4)]
                                            ]

                                            # Get authorized users from this zone
                                            if db_manager:
                                                try:
                                                    users_in_zone = db_manager.get_users_by_zone(zone_id)
                                                    zone['authorized_ids'] = [u.global_id for u in users_in_zone]
                                                except Exception as e:
                                                    logger.warning(f"Could not load users for zone: {e}")

                                        st.info(f"✅ Zone loaded from database: {len(zone.get('authorized_ids', []))} authorized users")

                                # Fetch users from database for dropdown
                                users_dict = fetch_users_dict()

                                if users_dict:
                                    # Create options for multiselect: "Name (ID: global_id)"
                                    user_options = {f"{name} (ID: {gid})": gid for gid, name in users_dict.items()}

                                    # Get current selected options
                                    current_selections = []
                                    for auth_id in zone['authorized_ids']:
                                        if auth_id in users_dict:
                                            current_selections.append(f"{users_dict[auth_id]} (ID: {auth_id})")

                                    logger.info(f"[Single-cam Zone {i+1}] current authorized_ids: {zone['authorized_ids']}, current_selections: {current_selections}")

                                    selected_users = st.multiselect(
                                        "Authorized Users",
                                        options=list(user_options.keys()),
                                        default=current_selections,
                                        key=f"zone_auth_{i}",
                                        help="Select authorized users from database"
                                    )

                                    logger.info(f"[Single-cam Zone {i+1}] selected_users from widget: {selected_users}")

                                    # Update authorized_ids based on selection
                                    # IMPORTANT: Only update if widget is not empty OR if current_selections was also empty
                                    # This prevents widget reset from clearing authorized_ids
                                    if selected_users or not current_selections:
                                        zone['authorized_ids'] = [user_options[user] for user in selected_users]
                                        logger.info(f"[Single-cam Zone {i+1}] final authorized_ids: {zone['authorized_ids']}")
                                    else:
                                        logger.warning(f"[Single-cam Zone {i+1}] Widget returned empty but current_selections was not empty - keeping existing authorized_ids: {zone['authorized_ids']}")
                                else:
                                    # Fallback to text input if database is not available
                                    st.warning("⚠️ Database not available. Using manual input.")
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

                                # Convert current polygon to string
                                current_polygon_str = '; '.join([f"{p[0]},{p[1]}" for p in zone['polygon']])

                                # Use widget key directly - Streamlit will manage the state
                                widget_key = f"zone_polygon_input_{i}"

                                # Initialize widget with current polygon value if not exists
                                if widget_key not in st.session_state:
                                    st.session_state[widget_key] = current_polygon_str

                                polygon_input = st.text_area(
                                    "Polygon Points",
                                    value=st.session_state[widget_key],
                                    key=widget_key,
                                    height=100,
                                    help="Enter coordinates as: x1,y1; x2,y2; x3,y3; ..."
                                )

                                # Parse and update zone polygon from user input
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

            # Helper function to create zones dict from zone list (DRY)
            def zones_list_to_dict(zones_list):
                """Convert list of zones to dict format for YAML."""
                zones_dict = {}
                for idx, zone in enumerate(zones_list):
                    zone_id = f"zone{idx+1}"
                    zones_dict[zone_id] = {
                        'name': zone['name'],
                        'polygon': zone['polygon'],
                        'authorized_ids': zone['authorized_ids']
                    }
                    logger.info(f"[YAML Export] Zone {zone_id} ({zone['name']}): authorized_ids={zone['authorized_ids']}")
                return zones_dict

            # Create YAML content from zones (always use cameras format)
            cameras_dict = {}

            if num_cameras > 1 and 'camera_zones' in st.session_state:
                # Multi-camera: use camera_zones
                for cam_idx in range(num_cameras):
                    camera_key = f'camera_{cam_idx+1}'
                    camera_zones = st.session_state.camera_zones.get(camera_key, [])

                    cameras_dict[camera_key] = {
                        'name': f'Camera {cam_idx+1}',
                        'zones': zones_list_to_dict(camera_zones)
                    }
            else:
                # Single camera: wrap in camera_1
                cameras_dict['camera_1'] = {
                    'name': 'Camera 1',
                    'zones': zones_list_to_dict(st.session_state.zones_config)
                }

            zones_data = {'cameras': cameras_dict}

            # Preview YAML
            if zones_data and zones_data.get('cameras'):
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

        # IoP Threshold (common for both methods)
        st.markdown("---")

        col_zone1, col_zone2 = st.columns(2)

        with col_zone1:
            iou_threshold = st.slider(
                "Zone IoP Threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="Percentage of person's body in zone (IoP) to detect. 60% recommended. Higher values (0.75-0.8) reduce false positives in overlapping zones."
            )

        with col_zone2:
            zone_opacity = st.slider(
                "Zone Border Thickness",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Zone border line thickness (0.0 = thin, 1.0 = thick). Controls border width from 1-10 pixels. 0.3 (3px) recommended."
            )

        # Zone processing threads
        st.markdown("### ⚙️ Zone Processing Performance")
        col_thread1, col_thread2 = st.columns(2)

        with col_thread1:
            zone_workers = st.number_input(
                "Zone Worker Threads",
                min_value=1,
                max_value=32,
                value=None,  # None = auto-detect (capped at 4)
                step=1,
                help="Number of threads for zone processing. None = auto-detect (capped at 4). Higher values = faster processing but more CPU usage."
            )
            if zone_workers is None:
                st.caption("🔄 Auto-detect (capped at 4 threads)")
            else:
                st.caption(f"🔧 Using {zone_workers} thread(s)")

        with col_thread2:
            st.info(
                "💡 **Thread Tips:**\n"
                "- **1 thread**: Low CPU, sequential processing\n"
                "- **2-4 threads**: Balanced (recommended)\n"
                "- **>4 threads**: High CPU, parallel processing"
            )

        # Alert threshold setting
        st.markdown("### 🚨 Violation Alert Settings")
        alert_threshold = st.number_input(
            "Alert Threshold (seconds)",
            min_value=0,
            max_value=10000,
            value=0,
            step=5,
            help="Time (in seconds) a person must be outside their authorized zone before triggering an alert. 0 = immediate alert."
        )

        # Livestream settings
        st.markdown("### 📡 Live Preview Settings")
        enable_livestream = st.checkbox(
            "Enable Live Preview",
            value=False,
            help="Enable real-time HLS livestream of AI-processed video (with bounding boxes, tracking, labels). View at http://localhost:3900"
        )

        if enable_livestream:
            st.info("📡 Live preview will be available at: **http://localhost:3900** during processing. The stream shows real-time AI detection with bounding boxes and tracking.")
            st.caption("💡 You can adjust buffer settings (segment duration, playlist size) in the livestream dashboard.")

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

            face_conf_thresh = st.slider(
                "Face Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Face detection confidence threshold (higher = stricter face detection)"
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
Face Detection: {face_conf_thresh}
Tracking: {track_thresh}
Zone Monitoring: {'Enabled' if zone_config_file else 'Disabled'}
IoP Threshold: {iou_threshold} ({iou_threshold*100:.0f}% of person in zone)
Zone Border Thickness: {int(zone_opacity*10)}px
            """)

    if st.button("🚀 Start Detection", type="primary"):
        logger.info(f"📌 [Detect & Track] Start Detection button clicked")

        # Validate input
        if input_method == "Upload Video File" and video_file is None:
            logger.warning(f"⚠️ [Detect & Track] No video file uploaded")
            st.error("Please upload a video file")
        elif input_method == "Stream URL (UDP/RTSP)" and not stream_url:
            logger.warning(f"⚠️ [Detect & Track] No stream URL provided")
            st.error("Please enter a stream URL")
        else:
            if input_method == "Upload Video File":
                logger.info(f"📹 [Detect & Track] Uploading video: {video_file.name} ({len(video_file.getvalue()) / (1024*1024):.2f} MB)")
            else:
                logger.info(f"📡 [Detect & Track] Stream URL: {stream_url}")
                if max_frames:
                    logger.info(f"⏱️ [Detect & Track] Max frames: {max_frames}")
                if max_duration:
                    logger.info(f"⏱️ [Detect & Track] Max duration: {max_duration}s")

            logger.info(f"⚙️ [Detect & Track] Parameters: model={model_type}, similarity={similarity_threshold}, conf={conf_thresh}, face_conf={face_conf_thresh}, track={track_thresh}")

            # Check if zone monitoring is enabled
            zone_enabled = zone_config_file is not None or zones_data is not None
            if zone_enabled:
                if zone_config_file:
                    logger.info(f"🗺️ [Detect & Track] Zone monitoring enabled (uploaded): {zone_config_file.name}")
                else:
                    # Count zones (always cameras format now)
                    total_zones = sum(len(cam_data['zones']) for cam_data in zones_data['cameras'].values())
                    num_cams = len(zones_data['cameras'])
                    if num_cams > 1:
                        logger.info(f"🗺️ [Detect & Track] Zone monitoring enabled (UI): {total_zones} zones across {num_cams} cameras")
                    else:
                        logger.info(f"🗺️ [Detect & Track] Zone monitoring enabled (UI): {total_zones} zones")

            spinner_text = "Uploading video and starting detection..." if input_method == "Upload Video File" else "Starting stream detection..."
            with st.spinner(spinner_text):
                try:
                    # Prepare files dict for multipart form data
                    files = {}

                    # Add zone config if provided
                    if zone_config_file:
                        files["zone_config"] = (zone_config_file.name, zone_config_file.getvalue(), "application/x-yaml")
                    elif zones_data:
                        yaml_content = yaml.dump(zones_data, default_flow_style=False, sort_keys=False)
                        files["zone_config"] = ("zones.yaml", yaml_content.encode('utf-8'), "application/x-yaml")

                    # Prepare common parameters (all as strings for multipart/form-data)
                    data = {
                        "similarity_threshold": str(similarity_threshold),
                        "iou_threshold": str(iou_threshold),
                        "zone_opacity": str(zone_opacity),
                        "alert_threshold": str(alert_threshold),
                        "enable_livestream": str(enable_livestream).lower()  # Convert bool to "true"/"false"
                    }

                    # Add optional parameters
                    if model_type:
                        data["model_type"] = model_type
                    if conf_thresh is not None:
                        data["conf_thresh"] = str(conf_thresh)
                    if track_thresh is not None:
                        data["track_thresh"] = str(track_thresh)
                    if face_conf_thresh is not None:
                        data["face_conf_thresh"] = str(face_conf_thresh)
                    if zone_enabled and zone_workers is not None:
                        data["zone_workers"] = str(zone_workers)

                    # Add input-specific parameters
                    if input_method == "Upload Video File":
                        # Add video file
                        files["video"] = (video_file.name, video_file.getvalue(), "video/mp4")
                        logger.info(f"🔄 [Detect & Track] Calling unified detection API with video file: {video_file.name}")
                    else:  # Stream URL
                        # Add stream URL and stream-specific parameters
                        data["stream_url"] = stream_url
                        if max_frames:
                            data["max_frames"] = str(max_frames)
                        if max_duration:
                            data["max_duration_seconds"] = str(max_duration)
                        logger.info(f"🔄 [Detect & Track] Calling unified detection API with stream URL: {stream_url}")
                        if max_frames:
                            logger.info(f"⏱️ [Detect & Track] Max frames: {max_frames}")
                        if max_duration:
                            logger.info(f"⏱️ [Detect & Track] Max duration: {max_duration}s")

                    # Call unified Detection API (handles both video files and streams)
                    response = requests.post(f"{DETECTION_API_URL}/detect", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]
                        logger.info(f"✅ [Detect & Track] Detection job created: {job_id}")

                        # Store job_id in session state for display after rerun
                        st.session_state['detect_current_job_id'] = job_id

                        st.info(f"Job ID: {job_id}")

                        # Show livestream player if enabled
                        if enable_livestream and 'livestream_url' in result:
                            livestream_url = result['livestream_url']
                            st.markdown("### 📡 Live Preview")
                            st.info(f"🎬 Livestream URL: {livestream_url}")

                            # Embed HLS player
                            hls_player_html = f"""
                            <div style="margin: 20px 0;">
                                <video id="video" controls autoplay muted style="width: 100%; max-width: 1200px; background: #000;"></video>
                                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                                <script>
                                    var video = document.getElementById('video');
                                    var videoSrc = '{livestream_url}';

                                    if (Hls.isSupported()) {{
                                        var hls = new Hls({{
                                            enableWorker: true,
                                            lowLatencyMode: true,
                                            backBufferLength: 90
                                        }});
                                        hls.loadSource(videoSrc);
                                        hls.attachMedia(video);
                                        hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                                            video.play();
                                        }});
                                        hls.on(Hls.Events.ERROR, function(event, data) {{
                                            if (data.fatal) {{
                                                console.error('HLS error:', data);
                                                setTimeout(function() {{
                                                    hls.loadSource(videoSrc);
                                                }}, 3000);
                                            }}
                                        }});
                                    }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                                        video.src = videoSrc;
                                        video.addEventListener('loadedmetadata', function() {{
                                            video.play();
                                        }});
                                    }}
                                </script>
                            </div>
                            """
                            st.components.v1.html(hls_player_html, height=600)
                            st.caption("📡 Latency: ~2-5 seconds | 🤖 Real-time AI detection with bounding boxes and tracking")

                        # Create placeholders for real-time updates
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        progress_text = st.empty()
                        tracks_container = st.empty()

                        # Initialize session state for WebSocket logs
                        if 'ws_logs' not in st.session_state:
                            st.session_state.ws_logs = []
                        if 'ws_connected' not in st.session_state:
                            st.session_state.ws_connected = False
                        if 'ws_queue' not in st.session_state:
                            st.session_state.ws_queue = Queue()

                        # WebSocket Realtime Violation Logs viewer
                        st.markdown("### 🔴 Realtime Violation Logs (WebSocket)")
                        st.caption("Live zone violation alerts as they occur")

                        ws_container = st.container()
                        with ws_container:
                            ws_status = st.empty()
                            ws_logs_display = st.empty()

                        # Kafka Realtime Alerts viewer
                        st.markdown("### 📡 Kafka Realtime Alerts")
                        st.caption("Realtime zone violation alerts streamed via Kafka")

                        kafka_alerts_container = st.container()
                        with kafka_alerts_container:
                            kafka_alerts_display = st.empty()
                            kafka_status = st.empty()

                        stop_button_container = st.empty()

                        poll_count = 0
                        user_cancelled = False
                        kafka_alert_lines = []  # Store Kafka alert lines for display
                        last_kafka_check = time.time()
                        kafka_consumer_url = "http://localhost:8004"

                        # Start WebSocket client thread (only once per job)
                        if not st.session_state.ws_connected and not hasattr(st.session_state, f'ws_thread_{job_id}'):
                            ws_message_queue = st.session_state.ws_queue
                            ws_thread = threading.Thread(
                                target=websocket_client_thread,
                                args=(job_id, ws_message_queue, DETECTION_API_URL),
                                daemon=True
                            )
                            ws_thread.start()
                            setattr(st.session_state, f'ws_thread_{job_id}', ws_thread)
                            logger.info(f"🔌 WebSocket client thread started for job {job_id}")

                        while True:
                            # Process WebSocket messages from queue
                            try:
                                while not st.session_state.ws_queue.empty():
                                    msg = st.session_state.ws_queue.get_nowait()

                                    if msg["type"] == "connection":
                                        if msg["status"] == "connected":
                                            st.session_state.ws_connected = True
                                            ws_status.success("✅ WebSocket Connected - Receiving realtime logs")
                                            logger.info(f"✅ WebSocket connected for job {job_id}")
                                        elif msg["status"] == "failed":
                                            ws_status.warning(f"⚠️ WebSocket connection failed: {msg.get('error', 'Unknown error')}")
                                            logger.warning(f"⚠️ WebSocket failed: {msg.get('error')}")
                                        else:
                                            ws_status.error(f"❌ WebSocket error: {msg.get('error', 'Unknown error')}")

                                    elif msg["type"] == "violation":
                                        violation_data = msg["data"]
                                        # Format: {timestamp, level, zone, message, frame}
                                        log_entry = f"[{violation_data.get('timestamp', '??:??:??')}] **{violation_data.get('zone', 'Unknown')}**: {violation_data.get('message', 'Unknown violation')} (Frame {violation_data.get('frame', 0)})"
                                        st.session_state.ws_logs.append(log_entry)

                                        # Keep only last 50 logs
                                        if len(st.session_state.ws_logs) > 50:
                                            st.session_state.ws_logs.pop(0)

                                        logger.debug(f"📨 WebSocket violation received: {violation_data.get('zone')} - {violation_data.get('message')}")
                            except Exception as e:
                                logger.debug(f"WebSocket queue error: {e}")

                            # Display WebSocket logs
                            if st.session_state.ws_logs:
                                logs_text = "\n\n".join(st.session_state.ws_logs[-10:])  # Show last 10 logs
                                ws_logs_display.markdown(logs_text)
                            elif st.session_state.ws_connected:
                                ws_logs_display.info("⏳ Waiting for zone violations...")

                            # Show stop button while processing
                            if stop_button_container.button("🛑 Stop Processing", type="secondary", key=f"stop_{job_id}_{poll_count}"):
                                logger.info(f"🛑 [Detect & Track] User requested to stop job: {job_id}")
                                try:
                                    cancel_response = requests.post(f"{DETECTION_API_URL}/cancel/{job_id}")
                                    if cancel_response.status_code == 200:
                                        logger.info(f"✅ [Detect & Track] Cancellation request sent for job: {job_id}")
                                        user_cancelled = True
                                        st.warning("⚠️ Stopping processing... Please wait.")
                                    else:
                                        logger.error(f"❌ [Detect & Track] Failed to cancel job: {cancel_response.text}")
                                        st.error(f"Failed to cancel job: {cancel_response.text}")
                                except Exception as e:
                                    logger.error(f"❌ [Detect & Track] Error cancelling job: {e}")
                                    st.error(f"Error cancelling job: {e}")
                            poll_count += 1
                            # Get progress
                            try:
                                progress_response = requests.get(f"{DETECTION_API_URL}/progress/{job_id}")
                                if progress_response.status_code == 200:
                                    progress = progress_response.json()

                                    # Check if multi-camera mode
                                    is_multi_camera = progress.get('cameras') is not None

                                    if is_multi_camera:
                                        # Multi-camera progress display
                                        cameras_data = progress['cameras']
                                        num_cameras = len(cameras_data)

                                        # Update progress bar (average across cameras)
                                        progress_bar.progress(0.5)  # Indeterminate for streams

                                        # Update status text
                                        status_text.text(f"Status: {progress['status']} ({num_cameras} cameras)")

                                        # Update progress text
                                        avg_frame = sum(cam.get('current_frame', 0) for cam in cameras_data.values()) // num_cameras
                                        progress_text.text(f"📊 Processing {num_cameras} cameras (avg frame: {avg_frame})")

                                        if poll_count % 10 == 0:
                                            logger.info(f"📊 [Detect & Track] Multi-camera progress: {num_cameras} cameras, avg frame {avg_frame}")

                                        # Display tracks per camera
                                        if progress['tracks']:
                                            tracks_info = "### 🎯 Current Tracks (All Cameras):\n"

                                            # Group tracks by camera
                                            tracks_by_camera = {}
                                            for track in progress['tracks']:
                                                cam_id = track.get('camera_id', 0)
                                                if cam_id not in tracks_by_camera:
                                                    tracks_by_camera[cam_id] = []
                                                tracks_by_camera[cam_id].append(track)

                                            # Display per camera
                                            for cam_id in sorted(tracks_by_camera.keys()):
                                                tracks_info += f"\n**📹 Camera {cam_id + 1}** (Frame {cameras_data[cam_id]['current_frame']}):\n"
                                                for track in tracks_by_camera[cam_id]:
                                                    color = "🟢" if track['label'] != "Unknown" else "🔴"
                                                    tracks_info += f"  {color} Track {track['track_id']}: **{track['label']}** (sim: {track['similarity']:.3f})\n"

                                            tracks_container.markdown(tracks_info)
                                    else:
                                        # Single-camera progress display (backward compatible)
                                        # Update progress bar
                                        if progress['total_frames'] > 0:
                                            progress_bar.progress(min(progress['progress_percent'] / 100, 0.99))
                                        else:
                                            # For streams, show indeterminate progress
                                            progress_bar.progress(0.5)

                                        # Update status text
                                        status_text.text(f"Status: {progress['status']}")

                                        # Update progress text
                                        if progress['total_frames'] > 0:
                                            progress_text.text(f"📊 Frame {progress['current_frame']}/{progress['total_frames']} ({progress['progress_percent']:.1f}%)")
                                        else:
                                            # For streams, show only current frame
                                            progress_text.text(f"📊 Frame {progress['current_frame']} (streaming...)")

                                        if poll_count % 10 == 0:  # Log every 10 polls
                                            if progress['total_frames'] > 0:
                                                logger.info(f"📊 [Detect & Track] Progress: {progress['current_frame']}/{progress['total_frames']} ({progress['progress_percent']:.1f}%)")
                                            else:
                                                logger.info(f"📊 [Detect & Track] Progress: Frame {progress['current_frame']} (streaming)")

                                        # Display current tracks
                                        if progress['tracks']:
                                            tracks_info = "### 🎯 Current Tracks:\n"
                                            for track in progress['tracks']:
                                                color = "🟢" if track['label'] != "Unknown" else "🔴"
                                                tracks_info += f"{color} Track {track['track_id']}: **{track['label']}** (sim: {track['similarity']:.3f})\n"
                                            tracks_container.markdown(tracks_info)

                                    # Fetch Kafka alerts from consumer service (every 2 seconds)
                                    current_time = time.time()
                                    if current_time - last_kafka_check >= 2.0:
                                        last_kafka_check = current_time
                                        try:
                                            # Check Kafka consumer service health
                                            health_response = requests.get(f"{kafka_consumer_url}/health", timeout=1)
                                            if health_response.status_code == 200:
                                                health_data = health_response.json()
                                                kafka_enabled = health_data.get('kafka_enabled', False)
                                                kafka_running = health_data.get('kafka_running', False)
                                                messages_received = health_data.get('messages_received', 0)

                                                # Update status
                                                if kafka_enabled and kafka_running:
                                                    kafka_status.success(f"✅ Kafka Connected | Messages: {messages_received}")
                                                elif kafka_enabled:
                                                    kafka_status.warning("⚠️ Kafka Enabled but not running")
                                                else:
                                                    kafka_status.info("ℹ️ Kafka Disabled (enable in config.yaml)")

                                                # Note: In production, you would use WebSocket for realtime updates
                                                # For now, we show the status and message count
                                                # The actual alerts are sent via Kafka and can be consumed by external systems

                                                if messages_received > 0:
                                                    kafka_alerts_display.info(
                                                        f"📊 **Kafka Alert System Active**\n\n"
                                                        f"Total alerts sent to Kafka: **{messages_received}**\n\n"
                                                        f"Alerts are being streamed to Kafka topic: `person_reid_alerts`\n\n"
                                                        f"**Alert Schema:**\n"
                                                        f"- user_id, user_name, camera_id, zone_id, zone_name\n"
                                                        f"- iop (Intersection over Person), threshold, status, timestamp\n\n"
                                                        f"💡 Connect your Kafka consumer to receive realtime alerts!"
                                                    )
                                                else:
                                                    kafka_alerts_display.info(
                                                        "⏳ Waiting for zone violations to generate Kafka alerts..."
                                                    )
                                            else:
                                                kafka_status.error("❌ Kafka Consumer Service not available")
                                        except requests.exceptions.RequestException:
                                            kafka_status.warning("⚠️ Kafka Consumer Service not reachable (http://localhost:8004)")
                                        except Exception as e:
                                            logger.debug(f"Kafka check error: {e}")
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
                                    # Clear stop button
                                    stop_button_container.empty()

                                    # Always fetch ZIP file (works for both single-stream and multi-stream)
                                    logger.info(f"📦 [Detect & Track] Fetching ZIP file for job: {job_id}")
                                    zip_url = f"{DETECTION_API_URL}/download/zip/{job_id}"
                                    zip_cache_key = f"detect_zip_{job_id}"
                                    zip_filename_key = f"detect_zip_filename_{job_id}"

                                    if zip_cache_key not in st.session_state:
                                        try:
                                            logger.info(f"📥 [Detect & Track] Fetching ZIP: {zip_url}")
                                            zip_response = requests.get(zip_url)
                                            if zip_response.status_code == 200:
                                                st.session_state[zip_cache_key] = zip_response.content
                                                # Extract filename from Content-Disposition header
                                                content_disposition = zip_response.headers.get('content-disposition', '')
                                                if 'filename=' in content_disposition:
                                                    filename = content_disposition.split('filename=')[1].strip('"')
                                                    st.session_state[zip_filename_key] = filename
                                                else:
                                                    # Fallback to default name
                                                    st.session_state[zip_filename_key] = f"{job_id}_results.zip"
                                                logger.info(f"✅ [Detect & Track] ZIP cached: {len(zip_response.content) / (1024*1024):.2f} MB, filename: {st.session_state[zip_filename_key]}")
                                            else:
                                                logger.error(f"❌ [Detect & Track] ZIP not available (status {zip_response.status_code})")
                                                st.error(f"Failed to fetch results ZIP (status {zip_response.status_code})")
                                        except Exception as e:
                                            logger.error(f"❌ [Detect & Track] Failed to fetch ZIP: {e}")
                                            st.error(f"Failed to fetch results: {e}")

                                    break

                                elif status["status"] == "failed":
                                    logger.error(f"❌ [Detect & Track] Detection failed: {status.get('error', 'Unknown error')}")
                                    st.error(f"❌ Detection failed: {status.get('error', 'Unknown error')}")
                                    # Clear stop button
                                    stop_button_container.empty()
                                    break

                                elif status["status"] == "cancelled":
                                    logger.info(f"🛑 [Detect & Track] Detection cancelled: {job_id}")
                                    st.warning("⚠️ Processing stopped by user")
                                    # Clear stop button
                                    stop_button_container.empty()
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

        # Get cached ZIP data
        zip_cache_key = f"detect_zip_{job_id}"
        zip_filename_key = f"detect_zip_filename_{job_id}"
        zip_data = st.session_state.get(zip_cache_key)
        zip_filename = st.session_state.get(zip_filename_key, f"{job_id}_results.zip")

        logger.info(f"📊 [Detect & Track] Cache status - ZIP: {bool(zip_data)}")

        # If ZIP not cached, fetch it now
        if not zip_data:
            logger.info(f"📦 [Detect & Track] ZIP not cached - fetching now")
            try:
                zip_url = f"{DETECTION_API_URL}/download/zip/{job_id}"
                zip_response = requests.get(zip_url)
                if zip_response.status_code == 200:
                    st.session_state[zip_cache_key] = zip_response.content
                    zip_data = zip_response.content
                    # Extract filename from Content-Disposition header
                    content_disposition = zip_response.headers.get('content-disposition', '')
                    if 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[1].strip('"')
                        st.session_state[zip_filename_key] = filename
                        zip_filename = filename
                    logger.info(f"✅ [Detect & Track] ZIP fetched and cached: {len(zip_data) / (1024*1024):.2f} MB")
                else:
                    logger.error(f"❌ [Detect & Track] Failed to fetch ZIP: status {zip_response.status_code}")
                    st.error(f"Failed to fetch results ZIP (status {zip_response.status_code})")
            except Exception as e:
                logger.error(f"❌ [Detect & Track] Failed to fetch ZIP: {e}")
                st.error(f"Failed to fetch results: {e}")

        # Show completion message
        if zip_data:
            st.success("✅ Processing completed! All outputs are ready for download.")
            st.info("📦 Download the ZIP file below to get all results (video, CSV, zone reports).")

        # Download ZIP file
        st.markdown("---")
        st.markdown("### 📦 Download Results")

        if zip_data:
            # Show ZIP content structure hint
            st.markdown("**ZIP File Contents:**")
            st.markdown("""
            - 📹 **Video**: Annotated output video with tracking boxes
            - 📊 **CSV**: Tracking data with person IDs and coordinates
            - 📋 **JSON**: Zone monitoring report (if enabled)

            For multi-camera jobs, files are organized by camera in separate folders.
            """)

            # Download button
            st.download_button(
                label="📦 Download ZIP",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True
            )
            logger.info(f"✅ [Detect & Track] ZIP download button displayed: {zip_filename}")
        else:
            st.warning("⚠️ Results not available. Please wait for processing to complete.")
elif page == "🗄️ DB Management":
    st.header("🗄️ Database Management")
    st.markdown("Manage users and working zones in PostgreSQL database")

    # Main sections: User Management and Zone Management
    main_tab1, main_tab2 = st.tabs(["👥 User Management", "📍 Zone Management"])

    # ============================================================================
    # USER MANAGEMENT SECTION
    # ============================================================================
    with main_tab1:
        # Tabs for different operations
        tab1, tab2, tab3 = st.tabs(["📋 View Users", "➕ Create User", "✏️ Edit/Delete User"])

        # Tab 1: View Users
        with tab1:
            st.subheader("All Users")

            if st.button("🔄 Refresh", key="refresh_users"):
                st.rerun()

            try:
                db_manager = get_db_manager()
                if db_manager:
                    users = db_manager.get_all_users()

                    if users:
                        # Display as table
                        import pandas as pd
                        users_data = [user.dict() for user in users]
                        df = pd.DataFrame(users_data)

                        # Select columns to display (including zone_id)
                        display_cols = ['id', 'global_id', 'name', 'zone_id', 'created_at', 'updated_at']
                        df_display = df[display_cols]

                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        st.success(f"✅ Total users: {len(users)}")
                    else:
                        st.info("No users found in database")
                else:
                    st.error("Failed to connect to database")
                    st.info("Check PostgreSQL connection settings in configs/.env")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Tab 2: Create User
        with tab2:
            st.subheader("Create New User")

            with st.form("create_user_form"):
                new_global_id = st.number_input("Global ID", min_value=1, value=1, help="Unique global ID for this user")
                new_name = st.text_input("Name", placeholder="e.g., John Doe")

                # Zone selection dropdown
                try:
                    db_manager = get_db_manager()
                    zones = db_manager.get_all_zones() if db_manager else []
                    zone_options = ["None (No Zone)"] + [f"{z.zone_name} ({z.zone_id})" for z in zones]
                    selected_zone_str = st.selectbox("Assign to Zone (Optional)", options=zone_options)

                    # Extract zone_id from selection
                    if selected_zone_str == "None (No Zone)":
                        selected_zone_id = None
                    else:
                        # Extract zone_id from "Zone Name (ZONE_ID)" format
                        selected_zone_id = selected_zone_str.split("(")[-1].rstrip(")")
                except Exception as e:
                    st.warning(f"Could not load zones: {e}")
                    selected_zone_id = None

                submitted = st.form_submit_button("➕ Create User", type="primary")

                if submitted:
                    if not new_name:
                        st.error("Please enter a name")
                    else:
                        try:
                            db_manager = get_db_manager()
                            if db_manager:
                                # Note: global_id can be duplicated, no need to check uniqueness
                                from services.database.models import UserCreate
                                user_data = UserCreate(
                                    global_id=new_global_id,
                                    name=new_name,
                                    zone_id=selected_zone_id
                                )
                                user = db_manager.create_user(user_data)

                                if user:
                                    st.success(f"✅ User created successfully! (ID: {user.id}, Global ID: {user.global_id})")
                                    st.json(user.dict())
                                    st.balloons()
                                else:
                                    st.error("Failed to create user")
                            else:
                                st.error("Database connection not available")
                        except Exception as e:
                            st.error(f"Error creating user: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        # Tab 3: Edit/Delete User
        with tab3:
            st.subheader("Edit or Delete User")

            try:
                db_manager = get_db_manager()
                if db_manager:
                    users = db_manager.get_all_users()

                    if users:
                        # Create selection dropdown
                        user_options = {f"{u.name} (ID: {u.global_id})": u for u in users}
                        selected_user_str = st.selectbox(
                            "Select User",
                            options=list(user_options.keys())
                        )

                        selected_user = user_options[selected_user_str]

                        st.divider()

                        # Edit section
                        st.markdown("### ✏️ Edit User")
                        with st.form("edit_user_form"):
                            edit_name = st.text_input("Name", value=selected_user.name)

                            # Zone selection dropdown for editing
                            zones = db_manager.get_all_zones()
                            zone_options = ["None (No Zone)"] + [f"{z.zone_name} ({z.zone_id})" for z in zones]

                            # Find current zone index
                            current_zone_idx = 0
                            if selected_user.zone_id:
                                for idx, opt in enumerate(zone_options):
                                    if selected_user.zone_id in opt:
                                        current_zone_idx = idx
                                        break

                            selected_zone_str = st.selectbox(
                                "Assign to Zone (Optional)",
                                options=zone_options,
                                index=current_zone_idx
                            )

                            # Extract zone_id from selection
                            if selected_zone_str == "None (No Zone)":
                                edit_zone_id = None
                            else:
                                edit_zone_id = selected_zone_str.split("(")[-1].rstrip(")")

                            submitted_edit = st.form_submit_button("💾 Update User", type="primary")

                            if submitted_edit:
                                try:
                                    from services.database.models import UserUpdate
                                    user_data = UserUpdate(name=edit_name, zone_id=edit_zone_id)
                                    updated_user = db_manager.update_user(selected_user.id, user_data)

                                    if updated_user:
                                        st.success("✅ User updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update user")
                                except Exception as e:
                                    st.error(f"Error updating user: {e}")

                        st.divider()

                        # Delete section
                        st.markdown("### 🗑️ Delete User")
                        st.warning(f"⚠️ You are about to delete: **{selected_user.name}** (Global ID: {selected_user.global_id})")

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("🗑️ Delete User", type="secondary"):
                                try:
                                    success = db_manager.delete_user(selected_user.id)

                                    if success:
                                        st.success("✅ User deleted successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete user")
                                except Exception as e:
                                    st.error(f"Error deleting user: {e}")
                    else:
                        st.info("No users found in database")
                else:
                    st.error("Database connection not available")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ============================================================================
    # ZONE MANAGEMENT SECTION
    # ============================================================================
    with main_tab2:
        # Tabs for different operations
        zone_tab1, zone_tab2, zone_tab3 = st.tabs(["📋 View Zones", "➕ Create Zone", "✏️ Edit/Delete Zone"])

        # Tab 1: View Zones
        with zone_tab1:
            st.subheader("All Working Zones")

            if st.button("🔄 Refresh", key="refresh_zones"):
                st.rerun()

            try:
                db_manager = get_db_manager()
                if db_manager:
                    zones = db_manager.get_all_zones()

                    if zones:
                        # Display as table with user count
                        import pandas as pd
                        zones_data = []
                        for zone in zones:
                            zone_dict = zone.dict()
                            # Get user count for this zone
                            users_in_zone = db_manager.get_users_by_zone(zone.zone_id)
                            zone_dict['user_count'] = len(users_in_zone)
                            zones_data.append(zone_dict)

                        df = pd.DataFrame(zones_data)

                        # Select columns to display (including user_count)
                        display_cols = ['zone_id', 'zone_name', 'user_count', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'created_at', 'updated_at']
                        df_display = df[display_cols]

                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        st.success(f"✅ Total zones: {len(zones)}")

                        # Show users in each zone
                        st.divider()
                        st.markdown("### 👥 Users in Zones")
                        for zone in zones:
                            users_in_zone = db_manager.get_users_by_zone(zone.zone_id)
                            with st.expander(f"📍 {zone.zone_name} ({zone.zone_id}) - {len(users_in_zone)} users"):
                                if users_in_zone:
                                    user_list = [f"- **{u.name}** (Global ID: {u.global_id})" for u in users_in_zone]
                                    st.markdown("\n".join(user_list))
                                else:
                                    st.info("No users assigned to this zone")
                    else:
                        st.info("No zones found in database")
                else:
                    st.error("Failed to connect to database")
                    st.info("Check PostgreSQL connection settings in configs/.env")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Tab 2: Create Zone
        with zone_tab2:
            st.subheader("Create New Working Zone")

            with st.form("create_zone_form"):
                new_zone_id = st.text_input("Zone ID", placeholder="e.g., ZONE_001", help="Unique zone identifier")
                new_zone_name = st.text_input("Zone Name", placeholder="e.g., Entrance Area")

                st.markdown("#### Zone Coordinates (4 points)")
                col1, col2 = st.columns(2)
                with col1:
                    x1 = st.number_input("X1", value=0.0, format="%.2f")
                    y1 = st.number_input("Y1", value=0.0, format="%.2f")
                    x2 = st.number_input("X2", value=0.0, format="%.2f")
                    y2 = st.number_input("Y2", value=0.0, format="%.2f")
                with col2:
                    x3 = st.number_input("X3", value=0.0, format="%.2f")
                    y3 = st.number_input("Y3", value=0.0, format="%.2f")
                    x4 = st.number_input("X4", value=0.0, format="%.2f")
                    y4 = st.number_input("Y4", value=0.0, format="%.2f")

                submitted = st.form_submit_button("➕ Create Zone", type="primary")

                if submitted:
                    if not new_zone_id or not new_zone_name:
                        st.error("Please enter zone ID and name")
                    else:
                        try:
                            db_manager = get_db_manager()
                            if db_manager:
                                # Check if zone_id already exists
                                existing_zone = db_manager.get_zone_by_id(new_zone_id)
                                if existing_zone:
                                    st.error(f"Zone with zone_id {new_zone_id} already exists: {existing_zone.zone_name}")
                                else:
                                    from services.database.models import WorkingZoneCreate
                                    zone_data = WorkingZoneCreate(
                                        zone_id=new_zone_id,
                                        zone_name=new_zone_name,
                                        x1=x1, y1=y1, x2=x2, y2=y2,
                                        x3=x3, y3=y3, x4=x4, y4=y4
                                    )
                                    zone = db_manager.create_zone(zone_data)

                                    if zone:
                                        st.success(f"✅ Zone created successfully!")
                                        st.json(zone.dict())
                                        st.balloons()
                                    else:
                                        st.error("Failed to create zone")
                            else:
                                st.error("Database connection not available")
                        except Exception as e:
                            st.error(f"Error creating zone: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        # Tab 3: Edit/Delete Zone
        with zone_tab3:
            st.subheader("Edit or Delete Working Zone")

            try:
                db_manager = get_db_manager()
                if db_manager:
                    zones = db_manager.get_all_zones()

                    if zones:
                        # Create selection dropdown
                        zone_options = {f"{z.zone_name} (ID: {z.zone_id})": z for z in zones}
                        selected_zone_str = st.selectbox(
                            "Select Zone",
                            options=list(zone_options.keys())
                        )

                        selected_zone = zone_options[selected_zone_str]

                        st.divider()

                        # Edit section
                        st.markdown("### ✏️ Edit Zone")
                        with st.form("edit_zone_form"):
                            edit_zone_name = st.text_input("Zone Name", value=selected_zone.zone_name)

                            st.markdown("#### Zone Coordinates (4 points)")
                            col1, col2 = st.columns(2)
                            with col1:
                                edit_x1 = st.number_input("X1", value=float(selected_zone.x1), format="%.2f")
                                edit_y1 = st.number_input("Y1", value=float(selected_zone.y1), format="%.2f")
                                edit_x2 = st.number_input("X2", value=float(selected_zone.x2), format="%.2f")
                                edit_y2 = st.number_input("Y2", value=float(selected_zone.y2), format="%.2f")
                            with col2:
                                edit_x3 = st.number_input("X3", value=float(selected_zone.x3), format="%.2f")
                                edit_y3 = st.number_input("Y3", value=float(selected_zone.y3), format="%.2f")
                                edit_x4 = st.number_input("X4", value=float(selected_zone.x4), format="%.2f")
                                edit_y4 = st.number_input("Y4", value=float(selected_zone.y4), format="%.2f")

                            submitted_edit = st.form_submit_button("💾 Update Zone", type="primary")

                            if submitted_edit:
                                try:
                                    from services.database.models import WorkingZoneUpdate
                                    zone_data = WorkingZoneUpdate(
                                        zone_name=edit_zone_name,
                                        x1=edit_x1, y1=edit_y1, x2=edit_x2, y2=edit_y2,
                                        x3=edit_x3, y3=edit_y3, x4=edit_x4, y4=edit_y4
                                    )
                                    updated_zone = db_manager.update_zone(selected_zone.zone_id, zone_data)

                                    if updated_zone:
                                        st.success("✅ Zone updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update zone")
                                except Exception as e:
                                    st.error(f"Error updating zone: {e}")

                        st.divider()

                        # Delete section
                        st.markdown("### 🗑️ Delete Zone")
                        st.warning(f"⚠️ You are about to delete: **{selected_zone.zone_name}** (Zone ID: {selected_zone.zone_id})")

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("🗑️ Delete Zone", type="secondary"):
                                try:
                                    success = db_manager.delete_zone(selected_zone.zone_id)

                                    if success:
                                        st.success("✅ Zone deleted successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete zone")
                                except Exception as e:
                                    st.error(f"Error deleting zone: {e}")
                    else:
                        st.info("No zones found in database")
                else:
                    st.error("Database connection not available")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# PAGE 5: ABOUT
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

