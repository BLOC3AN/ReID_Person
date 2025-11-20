# Kafka Realtime Alerts - Zone Monitoring

## Overview

Khi service detection chạy với zone monitoring, hệ thống sẽ **tự động publish realtime alerts lên Kafka topic** khi có zone violations.

## Architecture

```
Detection Service
    ↓
Zone Monitoring Service (Thread Pool)
    ↓
Zone Violation Detection
    ↓
Kafka Producer (send_alert)
    ↓
Kafka Topic: person_reid_alerts
    ↓
Kafka Consumer Service (WebSocket)
    ↓
UI (WebSocket clients)
```

## Configuration

### 1. Enable Kafka in `configs/config.yaml`

```yaml
kafka:
  enable: true                          # Enable/disable Kafka alerts
  bootstrap_servers: localhost:9092     # Kafka broker address
  topic: person_reid_alerts             # Topic name for alerts
  alert_threshold: 0.0                  # Time threshold before triggering alert (0 = immediate)
```

### 2. Kafka Config Loading Logic

**File:** `scripts/zone_monitor.py` (lines 865-887)

```python
# Load Kafka configuration from config
kafka_config = None
config_file_to_use = reid_config_path

# If no config path provided, use default config
if not config_file_to_use:
    from pathlib import Path
    default_config = Path(__file__).parent.parent / "configs" / "config.yaml"
    if default_config.exists():
        config_file_to_use = str(default_config)

if config_file_to_use:
    try:
        import yaml
        with open(config_file_to_use, 'r') as f:
            config = yaml.safe_load(f)
            if 'kafka' in config and config['kafka'].get('enable', False):
                kafka_config = config['kafka']
                logger.info(f"✅ Kafka enabled: {kafka_config.get('bootstrap_servers')} -> {kafka_config.get('topic')}")
```

**Key Point:** Kafka config được load từ default config file ngay cả khi `reid_config_path` là `None`.

## Alert Publishing Flow

### 1. Zone Service Initialization

**File:** `core/zone_service.py` (lines 100-114)

```python
# Kafka Producer
self.kafka_producer = None
if KAFKA_AVAILABLE and kafka_config and kafka_config.get('enable', False):
    try:
        self.kafka_producer = KafkaAlertProducer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            topic=kafka_config.get('topic', 'person_reid_alerts'),
            enable=True
        )
        logger.info(f"✅ Kafka Producer enabled for zone alerts (threshold: {alert_threshold}s)")
```

### 2. Violation Detection & Alert Publishing

**File:** `core/zone_service.py` (lines 301-381)

When a zone violation is detected:

```python
# Send Kafka alert for each missing person
if self.kafka_producer:
    for pid, name in zip(missing_persons, missing_names):
        self.kafka_producer.send_alert(
            user_id=str(pid),
            user_name=name,
            camera_id=camera_idx,
            zone_id=zone_id,
            zone_name=zone_state['name'],
            iop=zone_data.get('iou_threshold', 0.6),
            threshold=self.alert_threshold,
            status='violation_incomplete',
            frame_id=frame_id,
            additional_data={
                'violation_duration': round(violation_duration, 2),
                'missing_count': len(missing_persons),
                'required_count': len(zone_state['required_persons'])
            }
        )
```

### 3. Alert Message Format

**File:** `utils/kafka_manager.py` (lines 74-140)

```json
{
  "timestamp": "2024-11-20T10:30:45.123Z",
  "user_id": "1",
  "user_name": "Duong",
  "camera_id": 0,
  "zone_id": 1,
  "zone_name": "Assembly Area",
  "iop": 0.6,
  "threshold": 0.0,
  "status": "violation_incomplete",
  "frame_id": 150,
  "additional_data": {
    "violation_duration": 2.5,
    "missing_count": 1,
    "required_count": 2
  }
}
```

## Consumer Service

**File:** `services/kafka_consumer_service.py`

- Listens to Kafka topic `person_reid_alerts`
- Broadcasts messages to WebSocket clients
- Buffers last 100 messages for new clients

## Testing

Run the test script:

```bash
python tests/test_kafka_zone_alerts.py
```

This will:
1. Load Kafka config from `configs/config.yaml`
2. Initialize Zone Service with Kafka Producer
3. Start Kafka Consumer
4. Submit a test zone task
5. Verify Kafka messages are received

