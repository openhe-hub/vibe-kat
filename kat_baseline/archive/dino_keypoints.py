#!/usr/bin/env python3
"""DINO-ViT keypoint extraction following KAT paper (Di Palo & Johns, RSS 2024).

Paper-faithful implementation:
  - DINO-ViT/8 with stride=4 (overlapping patches, 128×128 feature map)
  - Layer 9 KEY features (not last-layer patch tokens)
  - Best Buddies Nearest Neighbours for salient descriptor selection
  - K=10 keypoints projected to 3D via calibrated RGBD camera

Pipeline (Fig. 3):
  1. Extract DINO-ViT dense key features from images
  2. Select K salient descriptors via Best Buddies NN
  3. Match descriptors in new images → 2D pixel coords
  4. Unproject to 3D via depth + camera calibration
"""

import os
import hashlib
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from camera_utils import batch_pixel_to_world


# ImageNet normalization (used by DINO)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoKeypointExtractor:
    """Extract 3D keypoints from images using DINO-ViT key features."""

    def __init__(self, model_name='dino_vitb8', stride=4, layer=9,
                 n_keypoints=10, device=None, cache_dir=None):
        """Initialize DINO-ViT model with paper-faithful parameters.

        Args:
            model_name: DINO model name ('dino_vitb8' or 'dino_vits8')
            stride: patch stride (paper uses 4 for overlapping patches)
            layer: which transformer layer to extract key features from (paper: 9)
            n_keypoints: number of keypoints K to extract (paper: 10)
            device: 'cuda' or 'cpu'
            cache_dir: directory for caching features
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.stride = stride
        self.layer = layer
        self.n_keypoints = n_keypoints
        self.cache_dir = cache_dir

        # Load DINO model
        print(f"Loading {model_name} (stride={stride}, layer={layer}) on {device}...")
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.eval()
        self.model.to(device)

        # Modify patch embedding stride for overlapping patches
        # Original: kernel_size=8, stride=8 → change stride to 4
        orig_stride = self.model.patch_embed.proj.stride
        if stride != orig_stride[0]:
            self.model.patch_embed.proj.stride = (stride, stride)
            print(f"  Changed patch stride: {orig_stride} → ({stride}, {stride})")

            # Monkey-patch interpolate_pos_encoding to handle new stride
            # The original uses w // patch_size which is wrong when stride != patch_size
            _orig_interp = self.model.interpolate_pos_encoding

            def _patched_interp(self_model, x, w, h):
                npatch = x.shape[1] - 1
                N = self_model.pos_embed.shape[1] - 1
                if npatch == N:
                    return self_model.pos_embed
                class_pos_embed = self_model.pos_embed[:, 0]
                patch_pos_embed = self_model.pos_embed[:, 1:]
                dim = x.shape[-1]
                # Compute actual grid size from number of patches
                import math
                w0 = h0 = int(math.sqrt(npatch))
                assert w0 * h0 == npatch, f"Non-square patch grid: {npatch} patches"
                # Original grid size
                orig_grid = int(math.sqrt(N))
                patch_pos_embed = patch_pos_embed.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
                patch_pos_embed = F.interpolate(
                    patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False
                )
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
                return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

            import types
            self.model.interpolate_pos_encoding = types.MethodType(_patched_interp, self.model)

        # Get model dimensions
        self.num_heads = self.model.blocks[0].attn.num_heads
        self.head_dim = self.model.embed_dim // self.num_heads
        self.feat_dim = self.model.embed_dim  # key features dim = embed_dim
        self.patch_size = self.model.patch_embed.proj.kernel_size[0]

        print(f"  embed_dim={self.model.embed_dim}, heads={self.num_heads}, "
              f"head_dim={self.head_dim}, feat_dim={self.feat_dim}")

        # Hook for extracting key features from specific layer
        self._key_features = None
        self._hook = self.model.blocks[layer - 1].attn.register_forward_hook(
            self._attn_hook
        )

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        print(f"DINO model loaded. Feature dim: {self.feat_dim}")

    def _attn_hook(self, module, input, output):
        """Forward hook to extract KEY features from attention layer.

        The attention module computes qkv = Linear(x), then splits into q, k, v.
        We extract the key features and reshape: (B, N, num_heads, head_dim) → (B, N, embed_dim).
        """
        x = input[0]  # (B, N_tokens, embed_dim)
        B, N, C = x.shape
        # qkv projection: (B, N, 3 * embed_dim)
        qkv = module.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        k = qkv[1]  # (B, heads, N, head_dim)
        # Concatenate heads: (B, N, heads * head_dim) = (B, N, embed_dim)
        k = k.permute(0, 2, 1, 3).reshape(B, N, -1)
        self._key_features = k

    def _image_hash(self, image):
        """SHA256 hash of image for caching."""
        return hashlib.sha256(image.tobytes()).hexdigest()[:16]

    @torch.no_grad()
    def extract_features(self, image):
        """Extract dense key features from an image.

        Args:
            image: (H, W, 3) uint8 RGB numpy array

        Returns:
            (Ph, Pw, D) feature array where Ph=Pw=H/stride (128 for 512px with stride 4)
        """
        # Check cache
        if self.cache_dir:
            h = self._image_hash(image)
            cache_path = os.path.join(self.cache_dir, f"s{self.stride}_l{self.layer}_{h}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)

        # Preprocess
        x = self.preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, 512, 512)
        H_img, W_img = x.shape[2], x.shape[3]
        Ph = (H_img - self.patch_size) // self.stride + 1
        Pw = (W_img - self.patch_size) // self.stride + 1

        # Forward pass — the hook captures key features
        _ = self.model(x)

        # Extract key features (hook stored them)
        # _key_features: (B, N_tokens, embed_dim) where N_tokens = 1 (CLS) + Ph*Pw
        keys = self._key_features[0, 1:, :]  # remove CLS token → (Ph*Pw, embed_dim)
        keys = keys.cpu().numpy()

        # Reshape to spatial grid
        feat_map = keys.reshape(Ph, Pw, self.feat_dim)

        # Cache
        if self.cache_dir:
            np.save(cache_path, feat_map)

        return feat_map

    def select_salient_descriptors(self, demo_features_list):
        """Select K salient descriptors via Best Buddies Nearest Neighbours.

        For each pair of demo images, find patches that are mutual nearest neighbors.
        Keep the K descriptors that appear most frequently across pairs.

        Args:
            demo_features_list: list of (Ph, Pw, D) feature arrays from demo images

        Returns:
            (K, D) array of salient descriptor vectors
        """
        n_demos = len(demo_features_list)

        if n_demos < 2:
            feat = demo_features_list[0]
            Ph, Pw, D = feat.shape
            flat = feat.reshape(-1, D)
            norms = np.linalg.norm(flat, axis=1)
            top_k = np.argsort(norms)[-self.n_keypoints:]
            return flat[top_k]

        # Flatten all feature maps
        flat_features = []
        for feat in demo_features_list:
            Ph, Pw, D = feat.shape
            flat_features.append(feat.reshape(-1, D))

        N = flat_features[0].shape[0]  # patches per image
        D = flat_features[0].shape[1]

        # Collect all best buddy descriptors across demo pairs
        all_buddy_descriptors = []

        for i in range(n_demos):
            for j in range(i + 1, n_demos):
                fi = flat_features[i]
                fj = flat_features[j]

                # Normalize for cosine similarity
                fi_norm = fi / (np.linalg.norm(fi, axis=1, keepdims=True) + 1e-8)
                fj_norm = fj / (np.linalg.norm(fj, axis=1, keepdims=True) + 1e-8)

                # For large feature maps (stride 4 → 16384 patches),
                # compute similarity in chunks to avoid OOM
                if N > 8000:
                    nn_i_to_j, nn_j_to_i = self._chunked_nn(fi_norm, fj_norm)
                else:
                    sim = fi_norm @ fj_norm.T
                    nn_i_to_j = np.argmax(sim, axis=1)
                    nn_j_to_i = np.argmax(sim, axis=0)

                # Mutual nearest neighbors (best buddies)
                for a in range(N):
                    b = nn_i_to_j[a]
                    if nn_j_to_i[b] == a:
                        all_buddy_descriptors.append(fi[a])

        if len(all_buddy_descriptors) == 0:
            flat = flat_features[0]
            norms = np.linalg.norm(flat, axis=1)
            top_k = np.argsort(norms)[-self.n_keypoints:]
            return flat[top_k]

        all_buddy_descriptors = np.array(all_buddy_descriptors)

        if len(all_buddy_descriptors) <= self.n_keypoints:
            remaining = self.n_keypoints - len(all_buddy_descriptors)
            flat = flat_features[0]
            norms = np.linalg.norm(flat, axis=1)
            top_extra = np.argsort(norms)[-remaining:]
            return np.vstack([all_buddy_descriptors, flat[top_extra]])

        descriptors = self._kmeans(all_buddy_descriptors, self.n_keypoints)
        return descriptors

    def _chunked_nn(self, fi_norm, fj_norm, chunk_size=4096):
        """Compute nearest neighbors in chunks to handle large feature maps."""
        N = fi_norm.shape[0]

        # i→j: for each row in fi, find best column in fj
        nn_i_to_j = np.zeros(N, dtype=int)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sim_chunk = fi_norm[start:end] @ fj_norm.T  # (chunk, N)
            nn_i_to_j[start:end] = np.argmax(sim_chunk, axis=1)

        # j→i: for each row in fj, find best column in fi
        nn_j_to_i = np.zeros(N, dtype=int)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sim_chunk = fj_norm[start:end] @ fi_norm.T  # (chunk, N)
            nn_j_to_i[start:end] = np.argmax(sim_chunk, axis=1)

        return nn_i_to_j, nn_j_to_i

    def _kmeans(self, data, k, max_iter=50):
        """Simple k-means clustering. Returns k centroids."""
        N, D = data.shape
        rng = np.random.RandomState(42)
        indices = rng.choice(N, size=k, replace=False)
        centroids = data[indices].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                members = data[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def find_keypoints_2d(self, image_features, salient_descriptors):
        """Find 2D pixel coordinates of K keypoints in an image.

        Args:
            image_features: (Ph, Pw, D) feature array
            salient_descriptors: (K, D) descriptor vectors

        Returns:
            (K, 2) pixel coordinates (u, v) — center of matching patch
        """
        Ph, Pw, D = image_features.shape
        flat = image_features.reshape(-1, D)

        flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        desc_norm = salient_descriptors / (np.linalg.norm(salient_descriptors, axis=1, keepdims=True) + 1e-8)

        sim = desc_norm @ flat_norm.T  # (K, N)
        best_patches = np.argmax(sim, axis=1)

        pixel_coords = np.zeros((len(salient_descriptors), 2))
        for i, patch_idx in enumerate(best_patches):
            row = patch_idx // Pw
            col = patch_idx % Pw
            # Pixel coordinate = patch center
            pixel_coords[i, 0] = col * self.stride + self.patch_size / 2  # u
            pixel_coords[i, 1] = row * self.stride + self.patch_size / 2  # v

        return pixel_coords

    def extract_keypoints_3d(self, image, depth, intrinsics, extrinsics_c2w,
                             salient_descriptors, near=None, far=None):
        """Full pipeline: image → DINO features → match → 3D keypoints.

        Args:
            image: (H, W, 3) uint8 RGB
            depth: (H, W) depth map
            intrinsics: (3, 3) camera intrinsic matrix
            extrinsics_c2w: (4, 4) camera-to-world matrix
            salient_descriptors: (K, D) from select_salient_descriptors
            near, far: depth clip planes (if depth is Z-buffer)

        Returns:
            (K, 3) world coordinates of keypoints
        """
        features = self.extract_features(image)
        pixel_coords = self.find_keypoints_2d(features, salient_descriptors)
        world_coords = batch_pixel_to_world(
            pixel_coords, depth, intrinsics, extrinsics_c2w,
            near=near, far=far
        )
        return world_coords
