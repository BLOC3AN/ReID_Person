#!/usr/bin/env python3
"""
Kafka Manager - Producer and Consumer for Alert System
Handles realtime alert messages with schema:
- user_id, user_name, camera_id, zone_id, zone_name, IOP, threshold, status, timestamp
"""

import json
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from loguru import logger
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import os


class KafkaAlertProducer:
    """
    Kafka Producer for sending alert messages
    Thread-safe and non-blocking
    """

    def __init__(self,
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'person_reid_alerts',
                 enable: bool = True):
        """
        Initialize Kafka Producer

        Args:
            bootstrap_servers: Kafka broker address
            topic: Topic name for alerts
            enable: Enable/disable Kafka (for testing without Kafka)
        """
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', bootstrap_servers)
        self.topic = topic
        self.enable = enable
        self.producer = None
        self.message_count = 0
        self.error_count = 0

        if not self.enable:
            logger.info("⚠️ Kafka Producer disabled (enable=False)")
            return

        try:
            # Producer configuration
            conf = {
                'bootstrap.servers': self.bootstrap_servers,
                'client.id': 'person_reid_alert_producer',
                'acks': 1,  # Wait for leader acknowledgment
                'compression.type': 'snappy',  # Compress messages
                'linger.ms': 10,  # Batch messages for 10ms
                'batch.size': 16384,  # Batch size in bytes
            }

            self.producer = Producer(conf)
            logger.info(f"✅ Kafka Producer initialized: {self.bootstrap_servers} -> topic '{self.topic}'")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Kafka Producer: {e}")
            self.enable = False

    def _delivery_callback(self, err, msg):
        """Callback when message is delivered or failed"""
        if err is not None:
            self.error_count += 1
            logger.error(f"❌ Kafka message delivery failed: {err}")
        else:
            self.message_count += 1
            logger.debug(f"✅ Kafka message delivered to {msg.topic()} [partition {msg.partition()}] at offset {msg.offset()}")

    def send_alert(self,
                   user_id: Optional[str],
                   user_name: str,
                   camera_id: int,
                   zone_id: int,
                   zone_name: str,
                   iop: float,
                   threshold: float,
                   status: str,
                   frame_id: Optional[int] = None,
                   additional_data: Optional[Dict] = None) -> bool:
        """
        Send alert message to Kafka

        Args:
            user_id: User global ID (None for Unknown)
            user_name: User name
            camera_id: Camera index
            zone_id: Zone ID
            zone_name: Zone name
            iop: Intersection over Person ratio (0.0-1.0)
            threshold: IOP threshold for zone
            status: Alert status (e.g., 'violation', 'incomplete', 'authorized')
            frame_id: Frame ID (optional)
            additional_data: Additional metadata (optional)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enable or self.producer is None:
            return False

        try:
            # Create message with schema
            message = {
                'user_id': user_id,
                'user_name': user_name,
                'camera_id': camera_id,
                'zone_id': zone_id,
                'zone_name': zone_name,
                'iop': round(iop, 3),
                'threshold': round(threshold, 3),
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'frame_id': frame_id,
            }

            # Add additional data if provided
            if additional_data:
                message.update(additional_data)

            # Serialize to JSON
            message_json = json.dumps(message)

            # Send to Kafka (non-blocking)
            self.producer.produce(
                topic=self.topic,
                value=message_json.encode('utf-8'),
                callback=self._delivery_callback
            )

            # Trigger delivery reports (non-blocking)
            self.producer.poll(0)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to send Kafka alert: {e}")
            self.error_count += 1
            return False

    def flush(self, timeout: float = 5.0):
        """
        Flush pending messages

        Args:
            timeout: Timeout in seconds
        """
        if self.producer:
            remaining = self.producer.flush(timeout)
            if remaining > 0:
                logger.warning(f"⚠️ {remaining} Kafka messages not delivered after flush")

    def close(self):
        """Close producer and flush pending messages"""
        if self.producer:
            logger.info(f"Closing Kafka Producer (sent: {self.message_count}, errors: {self.error_count})")
            self.flush()
            self.producer = None

    def get_stats(self) -> Dict[str, int]:
        """Get producer statistics"""
        return {
            'messages_sent': self.message_count,
            'errors': self.error_count,
            'enabled': self.enable
        }


class KafkaAlertConsumer:
    """
    Kafka Consumer for receiving alert messages
    Supports callback-based processing
    Auto-creates topic if it doesn't exist
    """

    def __init__(self,
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'person_reid_alerts',
                 group_id: str = 'person_reid_alert_consumers',
                 enable: bool = True):
        """
        Initialize Kafka Consumer

        Args:
            bootstrap_servers: Kafka broker address
            topic: Topic name for alerts
            group_id: Consumer group ID
            enable: Enable/disable Kafka
        """
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', bootstrap_servers)
        self.topic = topic
        self.group_id = group_id
        self.enable = enable
        self.consumer = None
        self.running = False
        self.message_count = 0

        if not self.enable:
            logger.info("⚠️ Kafka Consumer disabled (enable=False)")
            return

        try:
            # Try to create topic if it doesn't exist
            self._ensure_topic_exists()

            # Consumer configuration
            conf = {
                'bootstrap.servers': self.bootstrap_servers,
                'group.id': self.group_id,
                'client.id': 'person_reid_alert_consumer',
                'auto.offset.reset': 'latest',  # Start from latest messages
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 1000,
            }

            self.consumer = Consumer(conf)
            self.consumer.subscribe([self.topic])
            logger.info(f"✅ Kafka Consumer initialized: {self.bootstrap_servers} <- topic '{self.topic}' (group: {self.group_id})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Kafka Consumer: {e}")
            self.enable = False

    def _ensure_topic_exists(self):
        """Create topic if it doesn't exist"""
        try:
            admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})

            # Create topic
            topic_list = [NewTopic(self.topic, num_partitions=1, replication_factor=1)]
            fs = admin_client.create_topics(topic_list, validate_only=False)

            # Wait for operation to finish
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    logger.info(f"✅ Topic '{self.topic}' created successfully")
                except Exception as e:
                    # Topic might already exist, which is fine
                    if "already exists" in str(e).lower() or "TOPIC_ALREADY_EXISTS" in str(e):
                        logger.info(f"ℹ️ Topic '{self.topic}' already exists")
                    else:
                        logger.warning(f"⚠️ Topic creation warning: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not ensure topic exists: {e}")

    def consume(self, callback: Callable[[Dict], None], timeout: float = 1.0):
        """
        Consume messages and call callback for each message
        Blocking call - run in separate thread

        Args:
            callback: Function to call with each message dict
            timeout: Poll timeout in seconds
        """
        if not self.enable or self.consumer is None:
            logger.warning("Kafka Consumer not enabled")
            return

        self.running = True
        logger.info(f"🔄 Kafka Consumer started (topic: {self.topic})")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=timeout)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition - not an error
                        continue
                    else:
                        logger.error(f"❌ Kafka consumer error: {msg.error()}")
                        continue

                try:
                    # Decode message
                    message_json = msg.value().decode('utf-8')
                    message_dict = json.loads(message_json)

                    self.message_count += 1
                    logger.debug(f"✅ Kafka message received: {message_dict.get('status')} - {message_dict.get('zone_name')}")

                    # Call callback
                    callback(message_dict)

                except Exception as e:
                    logger.error(f"❌ Error processing Kafka message: {e}")

        except KeyboardInterrupt:
            logger.info("Kafka Consumer interrupted by user")
        except Exception as e:
            logger.error(f"❌ Kafka Consumer error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop consumer"""
        self.running = False
        if self.consumer:
            logger.info(f"Closing Kafka Consumer (received: {self.message_count} messages)")
            self.consumer.close()
            self.consumer = None

    def get_stats(self) -> Dict[str, int]:
        """Get consumer statistics"""
        return {
            'messages_received': self.message_count,
            'enabled': self.enable,
            'running': self.running
        }


# Singleton instances (optional - for easy access)
_global_producer: Optional[KafkaAlertProducer] = None
_global_consumer: Optional[KafkaAlertConsumer] = None


def get_kafka_producer(bootstrap_servers: str = 'localhost:9092',
                       topic: str = 'person_reid_alerts',
                       enable: bool = True) -> KafkaAlertProducer:
    """Get or create global Kafka producer instance"""
    global _global_producer
    if _global_producer is None:
        _global_producer = KafkaAlertProducer(bootstrap_servers, topic, enable)
    return _global_producer


def get_kafka_consumer(bootstrap_servers: str = 'localhost:9092',
                       topic: str = 'person_reid_alerts',
                       group_id: str = 'person_reid_alert_consumers',
                       enable: bool = True) -> KafkaAlertConsumer:
    """Get or create global Kafka consumer instance"""
    global _global_consumer
    if _global_consumer is None:
        _global_consumer = KafkaAlertConsumer(bootstrap_servers, topic, group_id, enable)
    return _global_consumer
