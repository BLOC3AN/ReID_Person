#!/usr/bin/env python3
"""
Livestream Service
Serves HLS livestream files for real-time viewing of AI-processed video
"""

from flask import Flask, send_from_directory, jsonify, render_template_string, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import logging
import os
import requests
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
HTTP_PORT = 3900
HLS_BASE_DIR = Path("/app/outputs/livestream")  # Base directory for all job livestreams
CONFIG_FILE = HLS_BASE_DIR / "config.json"  # HLS configuration file

# Load or create default config
def load_config():
    """Load HLS configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    # Default config
    return {
        "hls_segment_time": 6,
        "hls_list_size": 10
    }

def save_config(config):
    """Save HLS configuration to file"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# HTML template for main index page
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Detection Livestream Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            justify-content: center;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            text-align: center;
        }

        .stat-card .number {
            font-size: 2em;
            font-weight: bold;
        }

        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .jobs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }

        .job-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }

        .job-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }

        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }

        .job-id {
            font-size: 0.9em;
            color: #666;
            font-family: 'Courier New', monospace;
        }

        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }

        .status-processing {
            background: #4CAF50;
            color: white;
        }

        .status-completed {
            background: #2196F3;
            color: white;
        }

        .status-unknown {
            background: #9E9E9E;
            color: white;
        }

        .job-info {
            margin: 15px 0;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            color: #555;
        }

        .info-label {
            font-weight: 600;
        }

        .info-value {
            color: #667eea;
            font-weight: bold;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }

        .btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            text-align: center;
            display: block;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }

        .btn-secondary:hover {
            background: #e0e0e0;
        }

        .empty-state {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 60px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .empty-state h2 {
            color: #333;
            margin-bottom: 15px;
        }

        .empty-state p {
            color: #666;
            font-size: 1.1em;
        }

        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s;
        }

        .refresh-btn:hover {
            transform: scale(1.1) rotate(180deg);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4CAF50;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 AI Detection Livestream Dashboard</h1>
            <p>Real-time monitoring of AI-processed video streams</p>

            <div class="stats">
                <div class="stat-card">
                    <div class="number">{{ jobs|length }}</div>
                    <div class="label">Total Jobs</div>
                </div>
                <div class="stat-card">
                    <div class="number">{{ jobs|selectattr('status', 'equalto', 'processing')|list|length }}</div>
                    <div class="label">Live Now</div>
                </div>
                <div class="stat-card">
                    <div class="number">{{ jobs|selectattr('status', 'equalto', 'completed')|list|length }}</div>
                    <div class="label">Completed</div>
                </div>
            </div>
        </div>

        {% if jobs %}
        <div class="jobs-grid">
            {% for job in jobs %}
            <div class="job-card" onclick="window.location.href='{{ job.player_url }}'">
                <div class="job-header">
                    <div class="job-id">
                        {% if job.status == 'processing' %}
                        <span class="live-indicator"></span>
                        {% endif %}
                        {{ job.job_id[:16] }}...
                    </div>
                    <span class="status-badge status-{{ job.status }}">
                        {{ job.status }}
                    </span>
                </div>

                <div class="job-info">
                    <div class="info-row">
                        <span class="info-label">📅 Created:</span>
                        <span class="info-value">{{ job.created_at }}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🎞️ Segments:</span>
                        <span class="info-value">{{ job.segment_count }}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">💾 Size:</span>
                        <span class="info-value">{{ job.total_size_mb }} MB</span>
                    </div>
                </div>

                <div class="action-buttons">
                    <a href="{{ job.player_url }}" class="btn btn-primary" onclick="event.stopPropagation()">
                        {% if job.status == 'processing' %}
                        🔴 Watch Live
                        {% else %}
                        ▶️ Play Recording
                        {% endif %}
                    </a>
                    <a href="{{ job.hls_url }}" class="btn btn-secondary" onclick="event.stopPropagation()">
                        📄 Playlist
                    </a>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="empty-state">
            <h2>📭 No Livestreams Available</h2>
            <p>Start a detection job with livestream enabled to see it here.</p>
            <p style="margin-top: 20px; color: #999;">
                Go to <strong>Streamlit UI (port 8501)</strong> → Enable "Live Preview" → Start Detection
            </p>
        </div>
        {% endif %}
    </div>

    <button class="refresh-btn" onclick="location.reload()" title="Refresh">
        🔄
    </button>

    <a href="/settings" style="position: fixed; bottom: 110px; right: 30px; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; font-size: 1.5em; cursor: pointer; box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3); transition: all 0.3s; display: flex; align-items: center; justify-content: center; text-decoration: none;" title="Settings">
        ⚙️
    </a>

    <script>
        // Auto-refresh every 5 seconds
        setTimeout(function() {
            location.reload();
        }, 5000);
    </script>
</body>
</html>
"""

# HTML template for HLS player
PLAYER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Detection Livestream - Job {{ job_id }}</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        #video {
            width: 100%;
            max-width: 1200px;
            display: block;
            margin: 20px auto;
            background: #000;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.loading { background: #ff9800; }
        .status.playing { background: #4CAF50; }
        .status.error { background: #f44336; }
        .info {
            text-align: center;
            color: #aaa;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 AI Detection Livestream</h1>
        <div class="info">Job ID: {{ job_id }}</div>
        <div id="status" class="status loading">⏳ Loading stream...</div>
        <video id="video" controls autoplay muted></video>
        <div class="info">
            <p>📡 Latency: ~2-5 seconds (HLS standard)</p>
            <p>🤖 Real-time AI detection with bounding boxes and tracking</p>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const status = document.getElementById('status');
        const streamUrl = '/hls/{{ job_id }}/stream.m3u8';

        if (Hls.isSupported()) {
            const hls = new Hls({
                enableWorker: true,
                lowLatencyMode: true,
                backBufferLength: 90
            });

            hls.loadSource(streamUrl);
            hls.attachMedia(video);

            hls.on(Hls.Events.MANIFEST_PARSED, function() {
                status.textContent = '✅ Stream connected - Playing';
                status.className = 'status playing';
                video.play();
            });

            hls.on(Hls.Events.ERROR, function(event, data) {
                if (data.fatal) {
                    status.textContent = '❌ Stream error: ' + data.type;
                    status.className = 'status error';

                    // Auto-retry after 3 seconds
                    setTimeout(() => {
                        status.textContent = '🔄 Retrying...';
                        status.className = 'status loading';
                        hls.loadSource(streamUrl);
                    }, 3000);
                }
            });
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
            // Native HLS support (Safari)
            video.src = streamUrl;
            video.addEventListener('loadedmetadata', function() {
                status.textContent = '✅ Stream connected - Playing';
                status.className = 'status playing';
                video.play();
            });
        } else {
            status.textContent = '❌ HLS not supported in this browser';
            status.className = 'status error';
        }
    </script>
</body>
</html>
"""

# HTML template for settings page
SETTINGS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Livestream Settings</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .settings-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .form-group input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }

        .form-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: pointer;
        }

        .form-group input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: pointer;
            border: none;
        }

        .value-display {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }

        .info-box {
            background: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        }

        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ddd;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            font-weight: 600;
            color: #555;
        }

        .info-value {
            color: #667eea;
            font-weight: bold;
        }

        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
            margin-right: 10px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }

        .btn-secondary:hover {
            background: #e0e0e0;
        }

        .success-message {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }

        .preset-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .preset-btn {
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }

        .preset-btn:hover {
            background: #667eea;
            color: white;
        }

        .preset-btn strong {
            display: block;
            margin-bottom: 5px;
        }

        .preset-btn small {
            font-size: 0.85em;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚙️ Livestream Settings</h1>
            <p>Configure HLS streaming parameters for optimal playback</p>
        </div>

        {% if saved %}
        <div class="success-message">
            ✅ Settings saved successfully! New jobs will use these settings.
        </div>
        {% endif %}

        <form method="POST" class="settings-card">
            <h2 style="margin-bottom: 20px; color: #333;">HLS Configuration</h2>

            <div class="form-group">
                <label>
                    Segment Duration
                    <span class="value-display" id="segment-value">{{ config.hls_segment_time }}s</span>
                </label>
                <input type="range" name="hls_segment_time" id="segment-time"
                       min="2" max="100" step="1" value="{{ config.hls_segment_time }}"
                       oninput="updateValues()">
                <small style="color: #666;">Larger segments = smoother playback but higher latency</small>
            </div>

            <div class="form-group">
                <label>
                    Playlist Size
                    <span class="value-display" id="playlist-value">{{ config.hls_list_size }} segments</span>
                </label>
                <input type="range" name="hls_list_size" id="playlist-size"
                       min="5" max="200" step="1" value="{{ config.hls_list_size }}"
                       oninput="updateValues()">
                <small style="color: #666;">More segments = larger buffer for stability</small>
            </div>

            <div class="info-box">
                <h3>📊 Current Configuration</h3>
                <div class="info-row">
                    <span class="info-label">Segment Duration:</span>
                    <span class="info-value" id="info-segment">{{ config.hls_segment_time }}s</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Playlist Size:</span>
                    <span class="info-value" id="info-playlist">{{ config.hls_list_size }} segments</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Total Buffer:</span>
                    <span class="info-value" id="info-buffer">{{ buffer_time }}s</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Expected Latency:</span>
                    <span class="info-value" id="info-latency">~{{ config.hls_segment_time * 2 }}-{{ config.hls_segment_time * 3 }}s</span>
                </div>
            </div>

            <h3 style="margin: 25px 0 15px 0; color: #333;">🎯 Quick Presets</h3>
            <div class="preset-buttons">
                <div class="preset-btn" onclick="applyPreset(2, 5)">
                    <strong>Low Latency</strong>
                    <small>2s × 5 = 10s buffer</small>
                </div>
                <div class="preset-btn" onclick="applyPreset(4, 8)">
                    <strong>Balanced</strong>
                    <small>4s × 8 = 32s buffer</small>
                </div>
                <div class="preset-btn" onclick="applyPreset(6, 10)">
                    <strong>Smooth ⭐</strong>
                    <small>6s × 10 = 60s buffer</small>
                </div>
                <div class="preset-btn" onclick="applyPreset(8, 15)">
                    <strong>Ultra Smooth</strong>
                    <small>8s × 15 = 120s buffer</small>
                </div>
            </div>

            <div style="margin-top: 30px;">
                <button type="submit" class="btn btn-primary">💾 Save Settings</button>
                <a href="/" class="btn btn-secondary">← Back to Dashboard</a>
            </div>
        </form>
    </div>

    <script>
        function updateValues() {
            const segmentTime = parseInt(document.getElementById('segment-time').value);
            const playlistSize = parseInt(document.getElementById('playlist-size').value);
            const bufferTime = segmentTime * playlistSize;

            document.getElementById('segment-value').textContent = segmentTime + 's';
            document.getElementById('playlist-value').textContent = playlistSize + ' segments';
            document.getElementById('info-segment').textContent = segmentTime + 's';
            document.getElementById('info-playlist').textContent = playlistSize + ' segments';
            document.getElementById('info-buffer').textContent = bufferTime + 's';
            document.getElementById('info-latency').textContent = '~' + (segmentTime * 2) + '-' + (segmentTime * 3) + 's';
        }

        function applyPreset(segment, playlist) {
            document.getElementById('segment-time').value = segment;
            document.getElementById('playlist-size').value = playlist;
            updateValues();
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main UI - List all available livestreams"""
    # Get all job directories
    jobs = []
    if HLS_BASE_DIR.exists():
        for job_dir in sorted(HLS_BASE_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if job_dir.is_dir():
                job_id = job_dir.name
                playlist_file = job_dir / "stream.m3u8"

                # Check if playlist exists
                if playlist_file.exists():
                    # Get job info from detection service
                    try:
                        import requests
                        response = requests.get(f"http://localhost:8003/status/{job_id}", timeout=2)
                        if response.status_code == 200:
                            job_info = response.json()
                            status = job_info.get('status', 'unknown')
                            created_at = job_info.get('created_at', 'N/A')
                        else:
                            status = 'unknown'
                            created_at = 'N/A'
                    except:
                        status = 'unknown'
                        created_at = 'N/A'

                    # Count segments
                    segments = list(job_dir.glob("*.ts"))
                    segment_count = len(segments)

                    # Get total size
                    total_size = sum(f.stat().st_size for f in segments) / (1024 * 1024)  # MB

                    jobs.append({
                        'job_id': job_id,
                        'status': status,
                        'created_at': created_at,
                        'segment_count': segment_count,
                        'total_size_mb': round(total_size, 2),
                        'player_url': f'/player/{job_id}',
                        'hls_url': f'/hls/{job_id}/stream.m3u8'
                    })

    return render_template_string(INDEX_TEMPLATE, jobs=jobs)


@app.route('/player/<job_id>')
def player(job_id):
    """Web player for a specific job"""
    return render_template_string(PLAYER_TEMPLATE, job_id=job_id)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page for HLS configuration"""
    if request.method == 'POST':
        # Save settings
        config = {
            'hls_segment_time': int(request.form.get('hls_segment_time', 6)),
            'hls_list_size': int(request.form.get('hls_list_size', 10))
        }
        save_config(config)
        return redirect(url_for('settings') + '?saved=1')

    # Load current config
    config = load_config()
    saved = request.args.get('saved') == '1'

    # Calculate buffer info
    buffer_time = config['hls_segment_time'] * config['hls_list_size']

    return render_template_string(SETTINGS_TEMPLATE,
                                 config=config,
                                 buffer_time=buffer_time,
                                 saved=saved)



@app.route('/hls/<job_id>/<path:filename>')
def serve_hls(job_id, filename):
    """Serve HLS files (playlist and segments) for a specific job"""
    job_dir = HLS_BASE_DIR / job_id

    if not job_dir.exists():
        logger.warning(f"Job directory not found: {job_dir}")
        return jsonify({"error": "Job not found"}), 404

    # Security: only allow .m3u8 and .ts files
    if not (filename.endswith('.m3u8') or filename.endswith('.ts')):
        logger.warning(f"Invalid file type requested: {filename}")
        return jsonify({"error": "Invalid file type"}), 400

    logger.debug(f"Serving HLS file: {job_id}/{filename}")
    return send_from_directory(job_dir, filename)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "livestream"})


@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to avoid 404 errors"""
    return '', 204


if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🎬 AI DETECTION LIVESTREAM SERVICE")
    logger.info("="*80)
    logger.info(f"Port: {HTTP_PORT}")
    logger.info(f"HLS Base Directory: {HLS_BASE_DIR}")
    logger.info(f"Player URL: http://localhost:{HTTP_PORT}/player/<job_id>")
    logger.info("="*80)

    # Create base directory if not exists
    HLS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Run Flask server
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)

