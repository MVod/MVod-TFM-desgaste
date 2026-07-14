from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from branches.base import BaseBranch
from evaluation.aggregation import aggregate_tool_score
from preprocessing.m2_preprocessing import load_tool_roi_cached
from preprocessing.synthetic_mask import generate_ideal_mask, detect_thread_boundary
from preprocessing.registration import register_mask
from preprocessing.scale_calibration import load_scale_factor, get_px_per_mm


class DeviationScorer:
    """Measures pixel-level deviation between predicted and ideal mask."""

    def compute(self, predicted: np.ndarray, ideal: np.ndarray) -> float:
        """
        wear_ratio = |predicted XOR ideal| / |ideal|
        Returns 0.0 for perfect match, approaches 1.0 for complete mismatch.
        """
        pred_bin = (predicted > 127).astype(np.uint8)
        ideal_bin = (ideal > 127).astype(np.uint8)
        deviation = np.abs(pred_bin.astype(np.int16) - ideal_bin.astype(np.int16))
        ideal_area = ideal_bin.sum()
        if ideal_area == 0:
            return 0.0
        return float(deviation.sum()) / float(ideal_area)


class _SegDataset(Dataset):
    def __init__(self, tool_ids, tool_index, transform, px_per_mm, mask_config):
        self.samples = []
        for tid in tool_ids:
            paths, label = tool_index[tid]
            if label != 0:
                continue  # train segmentation only on normal images
            for p in paths:
                self.samples.append(p)
        self.transform = transform
        self.px_per_mm = px_per_mm
        self.mask_config = mask_config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        roi_mask, tool_roi = load_tool_roi_cached(path)
        H, W = roi_mask.shape

        boundary = detect_thread_boundary(tool_roi)
        center_row = H - 1 - boundary
        ideal = np.flipud(generate_ideal_mask(
            px_per_mm=self.px_per_mm,
            image_shape=(H, W),
            pitch_mm=self.mask_config["pitch_mm"],
            thread_height_mm=self.mask_config["thread_height_mm"],
            crest_width_mm=self.mask_config["crest_width_mm"],
            valley_width_mm=self.mask_config["valley_width_mm"],
            center_row=center_row,
        ))
        registered, _ = register_mask(tool_roi, ideal, roi_mask)
        gt = (registered > 127).astype(np.float32)

        img_t = self.transform(tool_roi)           # (1, H, W)
        gt_t = torch.from_numpy(gt).unsqueeze(0)   # (1, H, W)
        # Resize gt to match img_t spatial size
        _, tH, tW = img_t.shape
        if gt_t.shape[-2:] != (tH, tW):
            gt_t = torch.nn.functional.interpolate(
                gt_t.unsqueeze(0), size=(tH, tW), mode="nearest"
            ).squeeze(0)
        return img_t, gt_t


class BranchE(BaseBranch):
    """U-Net with EfficientNet-B0 encoder trained on normal images with synthetic masks."""

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._config: Dict = {}
        self._transform: Optional[T.Compose] = None
        self._px_per_mm: float = 348.0
        self._scorer = DeviationScorer()
        self._mask_config: Dict = {}

    def _build_transform(self, image_size: List[int]) -> T.Compose:
        H, W = image_size
        return T.Compose([
            T.ToPILImage(),
            T.Resize((H, W)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def train(self, train_tool_ids: List[str], tool_index: Dict, config: Dict) -> None:
        self._config = config
        self._px_per_mm = get_px_per_mm(config)
        image_size = config.get("image_size", [768, 1024])
        self._transform = self._build_transform(image_size)

        mask_config = {
            "pitch_mm": config.get("pitch_mm", 1.5),
            "thread_height_mm": config.get("thread_height_mm", 0.76),
            "crest_width_mm": config.get("crest_width_mm", 0.46),
            "valley_width_mm": config.get("valley_width_mm", 0.19),
        }
        self._mask_config = mask_config

        self._model = smp.Unet(
            encoder_name=config.get("backbone", "efficientnet-b0"),
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation="sigmoid",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        dataset = _SegDataset(
            train_tool_ids, tool_index, self._transform, self._px_per_mm, mask_config
        )
        if len(dataset) == 0:
            return  # no normal images → skip training

        loader = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 2),
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config.get("lr", 1e-3),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.get("epochs", 30)), eta_min=1e-5
        )
        criterion = nn.BCELoss()

        self._model.train()
        for epoch in range(config.get("epochs", 30)):
            for imgs, masks_gt in loader:
                imgs, masks_gt = imgs.to(device), masks_gt.to(device)
                optimizer.zero_grad()
                preds = self._model(imgs)
                loss = criterion(preds, masks_gt)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict(
        self,
        tool_id: str,
        tool_index: Dict,
        config: Dict,
    ) -> Tuple[float, List[float]]:
        assert self._model is not None, "Call train() before predict()"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        self._model.to(device)

        mask_config = self._mask_config
        image_scores = []
        with torch.no_grad():
            for img_path in tool_index[tool_id][0]:
                roi_mask, tool_roi = load_tool_roi_cached(img_path)
                H, W = roi_mask.shape

                boundary = detect_thread_boundary(tool_roi)
                center_row = H - 1 - boundary
                ideal = np.flipud(generate_ideal_mask(
                    px_per_mm=self._px_per_mm,
                    image_shape=(H, W),
                    center_row=center_row,
                    **mask_config,
                ))
                registered_ideal, _ = register_mask(tool_roi, ideal, roi_mask)

                tensor = self._transform(tool_roi).unsqueeze(0).to(device)
                pred = self._model(tensor).squeeze().cpu().numpy()
                pred_mask = (pred > 0.5).astype(np.uint8) * 255

                # Resize registered_ideal to match pred spatial dims after transform
                tH, tW = pred_mask.shape
                ideal_resized = cv2.resize(
                    registered_ideal, (tW, tH), interpolation=cv2.INTER_NEAREST
                )
                ratio = self._scorer.compute(pred_mask, ideal_resized)
                image_scores.append(ratio)
                if config.get("vis_dir"):
                    from pathlib import Path as _P
                    from visualization import save_mask_b
                    save_mask_b(img_path, tool_roi, pred_mask, ideal_resized,
                                ratio, _P(config["vis_dir"]), tool_id)

        tool_score = aggregate_tool_score(
            image_scores, method=config.get("aggregation", "mean")
        )
        # Clamp tool_score to [0, 1]
        tool_score = min(max(tool_score, 0.0), 1.0)
        # Clamp each image score to [0, 1]; deviation ratio is already bounded
        norm = [min(max(s, 0.0), 1.0) for s in image_scores]
        return tool_score, norm

    def save(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": self._config,
                "px_per_mm": self._px_per_mm,
                "mask_config": self._mask_config,
            },
            path,
        )

    def load(self, path: Path) -> None:
        # weights_only=False required: checkpoint contains non-tensor config dict.
        # Safe here because checkpoints are always self-generated by this pipeline.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._config = checkpoint["config"]
        self._px_per_mm = checkpoint["px_per_mm"]
        self._mask_config = checkpoint.get("mask_config", {})
        self._transform = self._build_transform(
            self._config.get("image_size", [768, 1024])
        )
        self._model = smp.Unet(
            encoder_name=self._config.get("backbone", "efficientnet-b0"),
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation="sigmoid",
        )
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
