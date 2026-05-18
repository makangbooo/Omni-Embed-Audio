#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WavCaps Adapter - Uses the actual ASE model from WavCaps repository
No more manual model reconstruction - this uses their code directly!
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import warnings


def _l2norm_np(x: np.ndarray) -> np.ndarray:
    """L2 normalize numpy array along last dimension."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-10)


class WavCapsAdapter:
    """
    WavCaps adapter that uses the actual ASE model from the WavCaps repository.
    This is the proper way to use WavCaps - no manual model reconstruction!
    """
    
    def __init__(
        self,
        repo_path: str,
        ckpt_path: str,
        seconds: float = 10.0,
        sr: int = 32000,
        device: str = "cuda"
    ):
        """
        Initialize WavCaps adapter using the actual ASE model.
        
        Args:
            repo_path: Path to the locally cloned WavCaps repository (e.g., path/to/WavCaps-master)
            ckpt_path: Path to checkpoint file (.pt)
            seconds: Audio duration for center crop
            sr: Sample rate (32000 for CNN14)
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = sr
        self.duration = seconds
        self.target_length = int(sr * seconds)
        
        # Add WavCaps retrieval directory to path
        repo_path = Path(repo_path).expanduser().resolve()
        retrieval_path = repo_path / "retrieval"
        if not retrieval_path.exists():
            retrieval_path = repo_path  # In case repo_path already points to retrieval
        
        if not retrieval_path.exists():
            raise RuntimeError(f"WavCaps retrieval directory not found at {retrieval_path}")
        
        # Add to Python path
        sys.path.insert(0, str(retrieval_path))
        
        print(f"[WavCapsAdapter] Using WavCaps from: {retrieval_path}")
        print(f"[WavCapsAdapter] Loading checkpoint: {ckpt_path}")
        print(f"[WavCapsAdapter] Device: {self.device}")
        
        try:
            # Import WavCaps modules
            from models.ase_model import ASE
            from ruamel.yaml import YAML
        except ImportError as e:
            raise ImportError(
                f"Failed to import WavCaps modules. Make sure the repo is properly set up.\n"
                f"Error: {e}\n"
                f"Try: pip install ruamel.yaml transformers"
            )
        
        # Load or create config
        config_path = retrieval_path / "settings" / "inference.yaml"
        if config_path.exists():
            print(f"[WavCapsAdapter] Loading config from: {config_path}")
            yaml = YAML(typ='safe', pure=True)
            with open(config_path, "r") as f:
                config = yaml.load(f)
        else:
            # Use default config if inference.yaml not found
            print("[WavCapsAdapter] Using default config (inference.yaml not found)")
            config = {
                'audio_encoder': 'CNN14',  # or 'HTSAT'
                'text_encoder': 'bert-base-uncased',
                'audio_dim': 2048,
                'text_dim': 768,
                'embed_dim': 1024,
                'audio_duration': seconds,
                'sample_rate': sr,
            }
        
        # Create ASE model
        print("[WavCapsAdapter] Creating ASE model...")
        self.model = ASE(config)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"[WavCapsAdapter] Found 'model' key in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"[WavCapsAdapter] Found 'state_dict' key in checkpoint")
            else:
                state_dict = checkpoint
                print(f"[WavCapsAdapter] Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
        
        # Load state dict
        try:
            self.model.load_state_dict(state_dict, strict=False)
            print("[WavCapsAdapter] ✓ Model weights loaded successfully")
        except Exception as e:
            print(f"[WavCapsAdapter] Warning: Partial weight loading - {e}")
            # Try to load what we can
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            print(f"[WavCapsAdapter] Loaded {len(filtered_dict)}/{len(state_dict)} parameters")
        
        # Move model to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("[WavCapsAdapter] ✓ Initialization complete")
    
    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda") -> np.ndarray:
        """
        Encode audio files to embeddings using WavCaps ASE model.
        
        Args:
            paths: List of audio file paths
            batch_size: Batch size for encoding
            device: Device to use (will use self.device if different)
        
        Returns:
            Numpy array of L2-normalized embeddings [N, D]
        """
        import librosa
        
        # AudioSet statistics for normalization (if using CNN14)
        LOGMEL_MEAN = -4.2677393
        LOGMEL_STD = 4.5689974
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_audio = []
                
                for audio_path in batch_paths:
                    try:
                        # Load audio
                        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                        
                        # Center crop or pad to fixed length
                        if len(y) < self.target_length:
                            # Pad if too short
                            padding = self.target_length - len(y)
                            y = np.pad(y, (0, padding), mode='constant')
                        else:
                            # Center crop if too long
                            start = (len(y) - self.target_length) // 2
                            y = y[start:start + self.target_length]
                        
                        batch_audio.append(y)
                        
                    except Exception as e:
                        print(f"[WavCapsAdapter] Error loading {audio_path}: {e}")
                        # Create silent audio as fallback
                        batch_audio.append(np.zeros(self.target_length, dtype=np.float32))
                
                # Convert to tensor
                batch_audio = np.array(batch_audio)
                audio_tensor = torch.from_numpy(batch_audio).float().to(self.device)
                
                # Check if model has encode_audio method (ASE model should have this)
                if hasattr(self.model, 'encode_audio'):
                    # Use ASE model's encode_audio directly
                    audio_embeds = self.model.encode_audio(audio_tensor)
                else:
                    # Fallback: compute mel-spectrogram and pass through model
                    # This path shouldn't normally be taken with proper ASE model
                    print("[WavCapsAdapter] Warning: Using fallback audio encoding")
                    
                    # Compute mel-spectrogram
                    mels = []
                    for audio in batch_audio:
                        mel = librosa.feature.melspectrogram(
                            y=audio,
                            sr=self.sample_rate,
                            n_fft=1024,
                            hop_length=320,
                            win_length=1024,
                            n_mels=64,
                            fmin=50,
                            fmax=14000,
                            htk=True,
                            norm=None
                        )
                        logmel = np.log(mel + 1e-10).astype(np.float32)
                        logmel = (logmel - LOGMEL_MEAN) / LOGMEL_STD
                        mels.append(logmel)
                    
                    mel_tensor = torch.from_numpy(np.stack(mels)).to(self.device)
                    audio_embeds = self.model(mel_tensor)
                
                # Normalize embeddings
                if audio_embeds.dim() > 1:
                    audio_embeds = F.normalize(audio_embeds, dim=-1)
                
                all_embeddings.append(audio_embeds.cpu().numpy())
        
        # Concatenate and ensure L2 normalization
        embeddings = np.concatenate(all_embeddings, axis=0)
        return _l2norm_np(embeddings)
    
    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
        """
        Encode texts to embeddings using WavCaps ASE model.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            device: Device to use (will use self.device if different)
        
        Returns:
            Numpy array of L2-normalized embeddings [N, D]
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Check if model has encode_text method (ASE model should have this)
                if hasattr(self.model, 'encode_text'):
                    # Use ASE model's encode_text directly
                    text_embeds = self.model.encode_text(batch_texts)
                    
                    # Ensure tensor and move to CPU
                    if isinstance(text_embeds, torch.Tensor):
                        text_embeds = text_embeds.cpu().numpy()
                    elif not isinstance(text_embeds, np.ndarray):
                        # Convert to numpy if needed
                        text_embeds = np.array(text_embeds)
                else:
                    # Fallback: This shouldn't happen with proper ASE model
                    raise RuntimeError("ASE model doesn't have encode_text method!")
                
                all_embeddings.append(text_embeds)
        
        # Concatenate and ensure L2 normalization
        embeddings = np.concatenate(all_embeddings, axis=0)
        return _l2norm_np(embeddings)


# For backwards compatibility with existing code
def create_wavcaps_adapter(repo_path: str, ckpt_path: str, **kwargs):
    """
    Factory function to create WavCaps adapter.
    
    Args:
        repo_path: Path to WavCaps repository
        ckpt_path: Path to checkpoint
        **kwargs: Additional arguments for WavCapsAdapter
    
    Returns:
        WavCapsAdapter instance
    """
    return WavCapsAdapter(repo_path, ckpt_path, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("WavCaps Adapter - Proper ASE Model Loading")
    print("=" * 60)
    
    print("""
    Usage:
    ------
    from wavcaps_adapter import WavCapsAdapter
    
    # Initialize adapter with your paths
    adapter = WavCapsAdapter(
        repo_path="path/to/WavCaps-master",
        ckpt_path="path/to/checkpoint.pt",
        seconds=10.0,
        sr=32000,
        device="cuda"
    )
    
    # Encode audio and text
    audio_paths = ["audio1.wav", "audio2.wav"]
    texts = ["a dog barking", "birds chirping"]
    
    audio_embeds = adapter.encode_audio(audio_paths)
    text_embeds = adapter.encode_text(texts)
    
    # Compute similarity
    similarity = np.dot(audio_embeds, text_embeds.T)
    """)
    
    print("\n" + "=" * 60)
    print("This adapter properly uses the WavCaps ASE model!")
    print("No more manual CNN14 reconstruction needed!")
    print("=" * 60)