# Stream Troubleshooting Guide

## Common Stream Issues and Solutions

### 1. "ffmpeg process terminated unexpectedly (exit code: 1)"

This error typically occurs when ffmpeg cannot connect to or decode the stream.

#### Possible Causes:
- **No stream broadcasting**: Nothing is actually sending data to the specified URL
- **Port already in use**: Another process is using the same port
- **Network connectivity**: Stream source is unreachable
- **Codec incompatibility**: Stream uses unsupported codec
- **Authentication**: Stream requires credentials

#### Solutions:

##### A. Check if stream is actually broadcasting
```bash
# Test with ffprobe
ffprobe -v quiet -show_entries format=duration udp://127.0.0.1:1905

# Test with ffplay (if available)
ffplay udp://127.0.0.1:1905
```

##### B. Use diagnostic tool
```bash
python debug_stream.py udp://127.0.0.1:1905
```

##### C. Check port availability
```bash
# Check if port is in use
netstat -an | grep 1905
lsof -i :1905
```

### 2. "Address already in use"

This error occurs when multiple processes try to bind to the same UDP port.

#### Solutions:
- **Stop conflicting processes**: Kill other processes using the port
- **Use different port**: Change stream URL to use available port
- **Enable port reuse**: StreamReader now automatically adds `reuse=1` parameter

### 3. Stream connection timeout

#### Symptoms:
- "No frame available from ffmpeg (3s timeout)"
- "ffmpeg returned 0 bytes"

#### Solutions:
- **Increase timeout**: Modify timeout values in StreamReader
- **Check network latency**: Ensure stable network connection
- **Verify stream format**: Some streams need specific parameters

### 4. Creating Test Stream

For testing purposes, you can create a local UDP stream:

#### Option 1: Using test simulator
```bash
# Create test video and start stream
python test_stream_simulator.py both 1905

# Or step by step
python test_stream_simulator.py create
python test_stream_simulator.py stream 1905
```

#### Option 2: Using ffmpeg directly
```bash
# Create test pattern and stream via UDP
ffmpeg -f lavfi -i testsrc=duration=60:size=640x480:rate=25 \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -f mpegts udp://127.0.0.1:1905
```

#### Option 3: Stream existing video file
```bash
# Stream video file in loop
ffmpeg -re -stream_loop -1 -i your_video.mp4 \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -f mpegts udp://127.0.0.1:1905
```

### 5. RTSP Streams

For RTSP streams, different issues may occur:

#### Common RTSP URLs:
```
rtsp://username:password@ip:port/stream
rtsp://192.168.1.100:554/stream1
```

#### RTSP-specific solutions:
- **Authentication**: Include credentials in URL
- **Transport protocol**: Try TCP instead of UDP
- **Timeout settings**: Increase connection timeout

### 6. Production Deployment

#### Docker Environment:
- Ensure proper network configuration
- Map required ports
- Check firewall settings

#### Network Configuration:
```bash
# Allow UDP traffic on specific port
sudo ufw allow 1905/udp

# Check network interface
ip addr show
```

### 7. Debugging Commands

#### Check ffmpeg capabilities:
```bash
ffmpeg -protocols | grep udp
ffmpeg -formats | grep mpegts
```

#### Monitor network traffic:
```bash
# Monitor UDP traffic on port 1905
sudo tcpdump -i any port 1905

# Check if packets are arriving
netstat -su | grep -i udp
```

#### Test with minimal ffmpeg command:
```bash
# Minimal test - just probe the stream
ffmpeg -timeout 5000000 -i udp://127.0.0.1:1905 -t 1 -f null -
```

### 8. StreamReader Configuration

The StreamReader class now includes:
- **Automatic fallback**: Falls back to OpenCV if ffmpeg fails
- **Port checking**: Validates UDP port availability
- **Better error messages**: Detailed diagnostic information
- **Improved timeout handling**: Configurable timeout values

#### Usage:
```python
from utils.stream_reader import StreamReader

# Create stream reader with automatic fallback
stream_reader = StreamReader("udp://127.0.0.1:1905", use_ffmpeg_for_udp=True)

# Get stream properties
props = stream_reader.get_properties()
print(f"Stream: {props['width']}x{props['height']} @ {props['fps']} FPS")

# Read frames
ret, frame = stream_reader.read()
if ret:
    print(f"Frame shape: {frame.shape}")
```

### 9. Performance Considerations

#### For high-performance streaming:
- **Pre-loaded pipeline**: Use pre-loaded components (already implemented)
- **Buffer management**: Adjust ffmpeg buffer sizes
- **Network optimization**: Use appropriate network settings
- **Hardware acceleration**: Consider GPU decoding if available

#### Monitoring:
- **Frame drops**: Monitor consecutive read failures
- **Latency**: Track processing time per frame
- **Memory usage**: Monitor buffer sizes

### 10. Contact and Support

If issues persist:
1. Run diagnostic tool: `python debug_stream.py <stream_url>`
2. Check logs for detailed error messages
3. Verify stream source is working with external tools
4. Test with different stream URLs/formats
