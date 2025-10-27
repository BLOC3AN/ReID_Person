#!/usr/bin/env python3
"""
Register Person from Video
Extract embeddings and register to database
"""

import sys
import cv2
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ByteTrack_Predict"))

from core import OSNetExtractor, QdrantVectorDB
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


class PersonRegistration:
    """Register person from video"""
    
    def __init__(self, config_path=None):
        """Initialize registration system"""
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("="*80)
        logger.info("Person Registration System")
        logger.info("="*80)
    
    def extract_frames(self, video_path, sample_rate=3):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")
        return frames
    
    def detect_person(self, frame, predictor):
        """Detect largest person bbox in frame"""
        outputs, img_info = predictor.inference(frame)
        
        if outputs[0] is None or len(outputs[0]) == 0:
            h, w = frame.shape[:2]
            return [0, 0, w, h]
        
        dets = outputs[0].cpu().numpy()
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        largest_idx = np.argmax(areas)
        det = dets[largest_idx]
        
        x1, y1, x2, y2 = det[:4]
        return [int(x1), int(y1), int(x2-x1), int(y2-y1)]
    
    def register(self, video_path, person_name, sample_rate=None):
        """
        Register person from video
        
        Args:
            video_path: Path to video file
            person_name: Name of person
            sample_rate: Extract 1 frame every N frames
        """
        if sample_rate is None:
            sample_rate = self.config['registration']['sample_rate']
        
        logger.info(f"Registering: {person_name}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Sample rate: {sample_rate}")
        
        # Initialize detector
        logger.info("\nInitializing YOLOX detector...")
        exp = get_exp(None, "yolox-x")
        model = exp.get_model()
        model.to(torch.device("cuda"))
        model.eval()
        
        ckpt_file = Path(__file__).parent.parent / "models" / "yolox_x.pth"
        if not ckpt_file.exists():
            # Try parent directory
            ckpt_file = Path(__file__).parent.parent.parent / "ByteTrack_Predict" / "pretrained" / "yolox_x.pth"
        
        ckpt = torch.load(str(ckpt_file), map_location="cuda")
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        
        class Predictor:
            def __init__(self, model, exp):
                self.model = model
                self.exp = exp
            
            def inference(self, frame):
                img, ratio = preproc(frame, self.exp.test_size, 
                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                with torch.no_grad():
                    outputs = self.model(torch.from_numpy(img[None]).cuda().float())
                outputs = postprocess(outputs, self.exp.num_classes, 
                                    self.exp.test_conf, self.exp.nmsthre)
                return outputs, {}
        
        predictor = Predictor(model, exp)
        
        # Initialize extractor
        logger.info("Initializing OSNet extractor...")
        extractor = OSNetExtractor(use_cuda=True)
        
        # Initialize database
        logger.info("Initializing database...")
        db = QdrantVectorDB(
            use_qdrant=self.config['database']['use_qdrant'],
            embedding_dim=self.config['database']['embedding_dim']
        )
        
        # Extract frames
        logger.info("\nExtracting frames...")
        frames = self.extract_frames(video_path, sample_rate)
        
        if len(frames) == 0:
            logger.error("No frames extracted!")
            return
        
        # Extract embeddings
        logger.info(f"\nExtracting embeddings from {len(frames)} frames...")
        embeddings = []
        
        for i, frame in enumerate(frames):
            bbox = self.detect_person(frame, predictor)
            embedding = extractor.extract(frame, bbox)
            if np.linalg.norm(embedding) > 0:
                embeddings.append(embedding)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(frames)} frames")
        
        logger.info(f"✅ Extracted {len(embeddings)} valid embeddings")
        
        if len(embeddings) == 0:
            logger.error("No valid embeddings extracted!")
            return
        
        # Register to database
        logger.info(f"\nRegistering {person_name} to database...")
        global_id = db.create_new_person(
            embeddings[0],
            metadata={'name': person_name, 'video_path': str(video_path)}
        )
        
        # Add embeddings (limit to max_frames)
        max_frames = self.config['registration']['max_frames']
        logger.info(f"Adding {min(len(embeddings), max_frames)} embeddings...")
        for embedding in embeddings[:max_frames]:
            db.add_embedding(global_id, embedding)
        
        # Save database
        db_dir = Path(__file__).parent.parent / "data" / "database"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / "reid_database.pkl"
        db.save_to_file(str(db_file))
        
        logger.info("\n" + "="*80)
        logger.info("✅ REGISTRATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Person: {person_name}")
        logger.info(f"Global ID: {global_id}")
        logger.info(f"Embeddings stored: {min(len(embeddings), max_frames)}")
        logger.info(f"Database: {db_file}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Register person from video")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--name", type=str, required=True, help="Person name")
    parser.add_argument("--sample-rate", type=int, default=None, 
                       help="Extract 1 frame every N frames")
    parser.add_argument("--config", type=str, default=None, 
                       help="Config file path")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Initialize registration
    registration = PersonRegistration(config_path=args.config)
    
    # Register person
    registration.register(
        video_path=video_path,
        person_name=args.name,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()

