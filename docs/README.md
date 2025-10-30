# Documentation Index

Welcome to the Person Re-Identification System documentation.

## Quick Links

### Getting Started
- **[Installation Guide](INSTALLATION.md)** - How to install and configure the system
- **[Deployment Guide](DEPLOYMENT.md)** - Docker deployment with GPU support
- **[Configuration Guide](CONFIGURATION.md)** - Complete configuration reference
- **[Usage Examples](USAGE.txt)** - Quick usage examples and commands

### Architecture & Design
- **[System Architecture](ARCHITECTURE.md)** - Microservices architecture and data flow
- **[Services API](SERVICES.md)** - RESTful API documentation for all services
- **[ReID Strategy](REID_STRATEGY.md)** - First-3 + Re-verify strategy details

### Reference
- **[API Documentation](API.md)** - Complete API reference with examples
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

### Package Information
- **[Package Manifest](MANIFEST.txt)** - Detailed package structure
- **[Package Info](PACKAGE_INFO.txt)** - Package details and contents
- **[Final Summary](FINAL_SUMMARY.txt)** - System verification summary

---

## Documentation Overview

### 1. Installation Guide (INSTALLATION.md)
Complete installation instructions including:
- System requirements
- Dependency installation (including InsightFace and ByteTrack dependencies)
- Qdrant configuration
- Verification steps
- Troubleshooting installation issues
- Note: ByteTrack is now fully integrated (yolox/ and exps/ directories)

### 2. Deployment Guide (DEPLOYMENT.md)
Docker deployment with GPU support:
- Local development setup
- Docker deployment with GPU
- Production deployment recommendations
- Monitoring and maintenance
- Troubleshooting deployment issues
- Performance tuning

### 3. Configuration Guide (CONFIGURATION.md)
Complete configuration reference:
- config.yaml detailed explanation
- .env file setup
- Parameter tuning for different use cases
- Best practices
- Advanced configuration options

### 4. System Architecture (ARCHITECTURE.md)
Microservices architecture and design:
- Core components overview
- Microservices architecture
- Data flow diagrams
- Storage architecture
- Performance optimization strategies
- Deployment architecture

### 5. Services API (SERVICES.md)
RESTful API documentation:
- Extract Service API (Port 8001)
- Register Service API (Port 8002)
- Detection Service API (Port 8003)
- UI Service (Port 8501)
- Error handling and best practices

### 6. Usage Examples (USAGE.txt)
Quick reference for common tasks:
- Extracting objects from multi-person videos
- Registering new persons (ArcFace face recognition)
- Running detection on videos
- Understanding output formats
- Parameter tuning
- Example workflows

### 7. API Documentation (API.md)
Comprehensive API reference:
- Core modules (Detector, Tracker, ArcFace Extractor, Database)
- Services API endpoints
- Script interfaces
- Configuration options
- Data formats
- Code examples

### 8. Troubleshooting Guide (TROUBLESHOOTING.md)
Solutions for common issues:
- Low similarity scores (ArcFace-specific)
- No face detected in bbox
- Qdrant connection problems
- CUDA/GPU issues
- Detection problems
- Performance optimization
- Debug mode

### 9. ReID Strategy (REID_STRATEGY.md)
First-3 + Re-verify strategy:
- Voting mechanism
- Re-verification logic
- Performance optimization
- Accuracy vs speed tradeoff

### 10. Package Manifest (MANIFEST.txt)
Detailed package structure:
- Folder hierarchy
- File descriptions
- Size information
- Registered persons
- System requirements

### 11. Package Info (PACKAGE_INFO.txt)
Package metadata:
- Version information
- Contents checklist
- Quick start guide
- Important notes

### 12. Final Summary (FINAL_SUMMARY.txt)
System verification:
- Package status
- Configuration verification
- Registered persons
- Demo outputs
- Production readiness

---

## Quick Start

### For Local Development
1. **Install:** Follow [INSTALLATION.md](INSTALLATION.md)
2. **Configure:** Setup [CONFIGURATION.md](CONFIGURATION.md)
3. **Extract Objects (Optional):** See [USAGE.txt](USAGE.txt)
4. **Register:** See [USAGE.txt](USAGE.txt)
5. **Detect:** See [USAGE.txt](USAGE.txt)
6. **Troubleshoot:** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### For Docker Deployment
1. **Setup GPU:** Follow [DEPLOYMENT.md](DEPLOYMENT.md) GPU setup section
2. **Configure:** Edit `configs/.env` with Qdrant credentials
3. **Deploy:** `cd deployment && sudo docker compose up -d`
4. **Access UI:** http://localhost:8501
5. **Monitor:** `sudo docker compose logs -f`

---

## Support

For detailed information on specific topics:

- **Installation issues** → [INSTALLATION.md](INSTALLATION.md)
- **Deployment with Docker** → [DEPLOYMENT.md](DEPLOYMENT.md)
- **Configuration tuning** → [CONFIGURATION.md](CONFIGURATION.md)
- **System architecture** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Services API** → [SERVICES.md](SERVICES.md)
- **How to use scripts** → [USAGE.txt](USAGE.txt)
- **API reference** → [API.md](API.md)
- **Error messages** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Package structure** → [MANIFEST.txt](MANIFEST.txt)

---

## Version

- **Version:** 2.0 (ArcFace Only)
- **Date:** 2025-10-28
- **Status:** Production Ready
- **Feature Extractor:** ArcFace (InsightFace) - Face Recognition

