# Documentation Index

Welcome to the Person Re-Identification System documentation.

## Quick Links

### Getting Started
- **[Installation Guide](INSTALLATION.md)** - How to install and configure the system
- **[Usage Examples](USAGE.txt)** - Quick usage examples and commands
- **[ArcFace Migration](ARCFACE_MIGRATION.md)** - 🆕 ArcFace vs OSNet comparison and migration guide

### Reference
- **[API Documentation](API.md)** - Complete API reference with examples
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[ReID Strategy](REID_STRATEGY.md)** - First-3 + Re-verify strategy details

### Package Information
- **[Package Manifest](MANIFEST.txt)** - Detailed package structure
- **[Package Info](PACKAGE_INFO.txt)** - Package details and contents
- **[Final Summary](FINAL_SUMMARY.txt)** - System verification summary

---

## Documentation Overview

### 1. Installation Guide (INSTALLATION.md)
Complete installation instructions including:
- System requirements
- Dependency installation (including InsightFace)
- Qdrant configuration
- Verification steps
- Troubleshooting installation issues

### 2. Usage Examples (USAGE.txt)
Quick reference for common tasks:
- Extracting objects from multi-person videos
- Registering new persons (with ArcFace)
- Running detection on videos
- Understanding output formats
- Parameter tuning
- Example workflows

### 3. ArcFace Migration Guide (ARCFACE_MIGRATION.md) 🆕
ArcFace vs OSNet comparison:
- What changed and why
- Pipeline comparison
- Configuration options
- When to use each extractor
- Migration from OSNet
- Performance comparison

### 4. API Documentation (API.md)
Comprehensive API reference:
- Core modules (Detector, Tracker, ArcFace/OSNet Extractor, Database)
- Script interfaces
- Configuration options
- Data formats
- Code examples

### 5. Troubleshooting Guide (TROUBLESHOOTING.md)
Solutions for common issues:
- Low similarity scores (ArcFace-specific)
- No face detected in bbox
- Qdrant connection problems
- CUDA/GPU issues
- Detection problems
- Performance optimization
- Switching between ArcFace and OSNet
- Debug mode

### 6. ReID Strategy (REID_STRATEGY.md)
First-3 + Re-verify strategy:
- Voting mechanism
- Re-verification logic
- Performance optimization
- Accuracy vs speed tradeoff

### 7. Package Manifest (MANIFEST.txt)
Detailed package structure:
- Folder hierarchy
- File descriptions
- Size information
- Registered persons
- System requirements

### 8. Package Info (PACKAGE_INFO.txt)
Package metadata:
- Version information
- Contents checklist
- Quick start guide
- Important notes

### 9. Final Summary (FINAL_SUMMARY.txt)
System verification:
- Package status
- Configuration verification
- Registered persons
- Demo outputs
- Production readiness

---

## Quick Start

1. **Install:** Follow [INSTALLATION.md](INSTALLATION.md)
2. **Extract Objects (Optional):** See [USAGE.txt](USAGE.txt) for extracting individual persons from multi-person videos
3. **Register:** See [USAGE.txt](USAGE.txt) for registration examples
4. **Detect:** See [USAGE.txt](USAGE.txt) for detection examples
5. **Troubleshoot:** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if issues occur

---

## Support

For detailed information on specific topics:

- **Installation issues** → [INSTALLATION.md](INSTALLATION.md)
- **How to use scripts** → [USAGE.txt](USAGE.txt)
- **API reference** → [API.md](API.md)
- **Error messages** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Package structure** → [MANIFEST.txt](MANIFEST.txt)

---

## Version

- **Version:** 1.1 (ArcFace Migration)
- **Date:** 2025-10-28
- **Status:** Production Ready
- **Feature Extractor:** ArcFace (InsightFace) - Face Recognition

