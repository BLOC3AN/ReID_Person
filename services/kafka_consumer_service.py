#!/usr/bin/env python3
"""
Kafka Consumer Service API
FastAPI service for consuming and streaming Kafka alerts to WebSocket clients
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import threading
from typing import Dict, List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
import json

# Import Kafka consumer
from utils.kafka_manager import KafkaAlertConsumer

app = FastAPI(title="Kafka Consumer Service", version="1.0.0")

# WebSocket connections
# {client_id: websocket}
websocket_clients: Dict[str, WebSocket] = {}

# Kafka consumer instance
kafka_consumer: KafkaAlertConsumer = None
consumer_thread: threading.Thread = None

# Message buffer for new clients (keep last 100 messages)
message_buffer: List[Dict] = []
MAX_BUFFER_SIZE = 100


def kafka_message_callback(message: Dict):
    """
    Callback when Kafka message is received
    Broadcast to all connected WebSocket clients
    """
    # Add to buffer
    message_buffer.append(message)
    if len(message_buffer) > MAX_BUFFER_SIZE:
        message_buffer.pop(0)
    
    # Broadcast to all clients
    asyncio.run(broadcast_message(message))


async def broadcast_message(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    
    for client_id, ws in list(websocket_clients.items()):
        try:
            await ws.send_json(message)
            logger.debug(f"✅ Sent Kafka alert to client {client_id}")
        except Exception as e:
            logger.warning(f"Failed to send to client {client_id}: {e}")
            disconnected.append(client_id)
    
    # Remove disconnected clients
    for client_id in disconnected:
        if client_id in websocket_clients:
            del websocket_clients[client_id]
            logger.info(f"Removed disconnected client: {client_id}")


@app.on_event("startup")
async def startup_event():
    """Start Kafka consumer on startup"""
    global kafka_consumer, consumer_thread
    
    logger.info("🚀 Starting Kafka Consumer Service...")
    
    # Initialize Kafka consumer
    kafka_consumer = KafkaAlertConsumer(
        bootstrap_servers='localhost:9092',
        topic='person_reid_alerts',
        group_id='person_reid_ui_consumers',
        enable=True
    )
    
    # Start consumer in background thread
    consumer_thread = threading.Thread(
        target=kafka_consumer.consume,
        args=(kafka_message_callback,),
        daemon=True,
        name="KafkaConsumerThread"
    )
    consumer_thread.start()
    
    logger.info("✅ Kafka Consumer Service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop Kafka consumer on shutdown"""
    global kafka_consumer
    
    logger.info("🛑 Stopping Kafka Consumer Service...")
    
    if kafka_consumer:
        kafka_consumer.stop()
    
    logger.info("✅ Kafka Consumer Service stopped")


@app.get("/")
async def root():
    return {"service": "Kafka Consumer Service", "status": "running"}


@app.get("/health")
async def health():
    stats = kafka_consumer.get_stats() if kafka_consumer else {}
    return {
        "status": "healthy",
        "kafka_enabled": stats.get('enabled', False),
        "kafka_running": stats.get('running', False),
        "messages_received": stats.get('messages_received', 0),
        "connected_clients": len(websocket_clients)
    }


@app.get("/stats")
async def stats():
    """Get service statistics"""
    kafka_stats = kafka_consumer.get_stats() if kafka_consumer else {}
    return {
        "kafka": kafka_stats,
        "websocket_clients": len(websocket_clients),
        "buffer_size": len(message_buffer)
    }


@app.get("/messages/recent")
async def get_recent_messages(limit: int = 10):
    """
    Get recent Kafka messages from buffer

    Args:
        limit: Number of recent messages to return (default: 10, max: 100)
    """
    limit = min(limit, MAX_BUFFER_SIZE)
    return {
        "messages": message_buffer[-limit:],
        "total_in_buffer": len(message_buffer)
    }


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming Kafka alerts
    Clients connect here to receive realtime alerts
    """
    await websocket.accept()
    
    # Generate client ID
    client_id = f"client_{id(websocket)}"
    websocket_clients[client_id] = websocket
    
    logger.info(f"✅ WebSocket client connected: {client_id} (total: {len(websocket_clients)})")
    
    try:
        # Send buffered messages to new client
        for msg in message_buffer:
            await websocket.send_json(msg)
        
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "message": f"Connected to Kafka alert stream (buffered: {len(message_buffer)} messages)",
            "timestamp": None
        })
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Remove client
        if client_id in websocket_clients:
            del websocket_clients[client_id]
        logger.info(f"Removed client {client_id} (remaining: {len(websocket_clients)})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

