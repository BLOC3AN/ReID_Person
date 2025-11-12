#!/bin/bash
# Start test UDP streams from video files (simpler than RTSP)

VIDEO_DIR="/home/ubuntu/data/person_reid_system/stores/videos"
VIDEO1="${VIDEO_DIR}/cam1.mkv"

# Check if video exists
if [ ! -f "$VIDEO1" ]; then
    echo "❌ Video not found: $VIDEO1"
    exit 1
fi

echo "🎬 Starting UDP test streams..."

# Kill existing streams
pkill -f "ffmpeg.*udp://127.0.0.1:190" 2>/dev/null
sleep 1

# Stream 1 to UDP port 1905
echo "📹 Starting stream 1 -> udp://127.0.0.1:1905"
ffmpeg -re -stream_loop -1 -i "$VIDEO1" \
    -c:v copy -an \
    -f mpegts udp://127.0.0.1:1905 \
    > /tmp/stream1.log 2>&1 &
STREAM1_PID=$!

# Stream 2 to UDP port 1906
echo "📹 Starting stream 2 -> udp://127.0.0.1:1906"
ffmpeg -re -stream_loop -1 -i "$VIDEO1" \
    -c:v copy -an \
    -f mpegts udp://127.0.0.1:1906 \
    > /tmp/stream2.log 2>&1 &
STREAM2_PID=$!

echo ""
echo "✅ Streams started!"
echo "   Stream 1: udp://127.0.0.1:1905 (PID: $STREAM1_PID)"
echo "   Stream 2: udp://127.0.0.1:1906 (PID: $STREAM2_PID)"
echo ""
echo "📊 Test with:"
echo "   ffplay udp://127.0.0.1:1905"
echo "   ffplay udp://127.0.0.1:1906"
echo ""
echo "🛑 Stop with:"
echo "   pkill -f 'ffmpeg.*udp://127.0.0.1:190'"
echo ""
echo "Logs: /tmp/stream1.log, /tmp/stream2.log"

# Wait for streams to start
sleep 2

# Test streams
echo ""
echo "🔍 Testing streams..."
timeout 3 ffprobe -v error udp://127.0.0.1:1905 2>&1 | grep -q "Stream" && echo "✅ Stream 1 OK" || echo "⏳ Stream 1 starting..."
timeout 3 ffprobe -v error udp://127.0.0.1:1906 2>&1 | grep -q "Stream" && echo "✅ Stream 2 OK" || echo "⏳ Stream 2 starting..."

echo ""
echo "💡 Streams are ready for benchmark!"

