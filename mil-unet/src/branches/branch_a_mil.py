from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from branches.base import BaseBranch
from evaluation.aggregation import aggregate_tool_score
from preprocessing.m2_preprocessing import load_tool_roi_cached


class MILClassifier(nn.Module):
    """EfficientNet-B0 backbone with a binary classification head."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,   # remove classifier head
            in_chans=3,
        )
        feat_dim = self.backbone.num_features  # 1280 for efficientnet_b0
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)   # (B, 1280)
        return self.head(feats)    # (B, 1)


class BranchA(BaseBranch):
    """MIL + EfficientNet-B0. Tool score = max(image scores) during training and inference."""

    def __init__(self) -> None:
        self._model: Optional[MILClassifier] = None
        self._config: Dict = {}
        self._transform: Optional[T.Compose] = None

    def _build_transform(self, image_size: List[int]) -> T.Compose:
        H, W = image_size
        return T.Compose([
            T.ToPILImage(),
            T.Resize((H, W)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _freeze_backbone(self, n_unfreeze_blocks: int = 0) -> None:
        for p in self._model.backbone.parameters():
            p.requires_grad_(False)
        if n_unfreeze_blocks > 0:
            blocks = list(self._model.backbone.blocks)[-n_unfreeze_blocks:]
            for block in blocks:
                for p in block.parameters():
                    p.requires_grad_(True)

    def _train_phase(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        grad_clip: float,
        train_tool_ids: List[str],
        tool_index: Dict,
        device: torch.device,
    ) -> None:
        """
        MIL training loop. Iterates tools; for each tool stacks all images,
        runs forward pass, applies max-pooling per tool, computes BCE loss.
        """
        criterion = nn.BCELoss()
        self._model.train()
        self._model.to(device)

        for epoch in range(epochs):
            for tool_id in train_tool_ids:
                paths, label = tool_index[tool_id]
                tensors = []
                for p in paths:
                    _, roi = load_tool_roi_cached(p)
                    t = self._transform(roi).to(device)
                    tensors.append(t)
                x = torch.stack(tensors)  # (n_imgs, 3, H, W)

                optimizer.zero_grad()
                scores = self._model(x)         # (n_imgs, 1)
                tool_score = scores.max()        # MIL max-pooling
                target = torch.tensor([[float(label)]], device=device)
                loss = criterion(tool_score.unsqueeze(0).unsqueeze(0), target)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                optimizer.step()

    def train(self, train_tool_ids: List[str], tool_index: Dict, config: Dict) -> None:
        self._config = config
        self._transform = self._build_transform(config.get("image_size", [384, 512]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = MILClassifier(dropout=config.get("dropout", 0.5))

        grad_clip = config.get("grad_clip", 1.0)

        # Phase 1: frozen backbone — only head trains
        self._freeze_backbone(n_unfreeze_blocks=0)
        opt1 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=config.get("lr_phase1", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self._train_phase(opt1, config.get("epochs_phase1", 10),
                          grad_clip, train_tool_ids, tool_index, device)

        # Phase 2: unfreeze last 2 backbone blocks for fine-tuning
        self._freeze_backbone(n_unfreeze_blocks=2)
        opt2 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=config.get("lr_phase2", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self._train_phase(opt2, config.get("epochs_phase2", 20),
                          grad_clip, train_tool_ids, tool_index, device)

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
        vis_dir = config.get("vis_dir")
        image_scores = []

        ctx = torch.enable_grad() if vis_dir else torch.no_grad()
        with ctx:
            for img_path in tool_index[tool_id][0]:
                roi_mask, roi = load_tool_roi_cached(img_path)
                tensor = self._transform(roi).unsqueeze(0).to(device)
                score = self._model(tensor).item()
                image_scores.append(score)
                if vis_dir:
                    self._save_gradcam(img_path, roi, roi_mask, tensor,
                                       score, vis_dir, config, tool_id)

        tool_score = aggregate_tool_score(
            image_scores, method=config.get("mil_pooling", "max")
        )
        return tool_score, image_scores

    def _save_gradcam(
        self,
        img_path,
        roi,
        roi_mask,
        tensor,
        score: float,
        vis_dir: str,
        config: Dict,
        tool_id: str,
    ) -> None:
        """Compute GradCAM++ map and save 4-panel visualization. Silently skips on error."""
        import warnings
        from pathlib import Path as _P
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from preprocessing.scale_calibration import get_px_per_mm
            from preprocessing.synthetic_mask import generate_ideal_mask, detect_thread_boundary
            from preprocessing.registration import register_mask
            from visualization import save_gradcam_a

            cam = GradCAMPlusPlus(
                model=self._model,
                target_layers=[self._model.backbone.conv_head],
            )
            gradcam_map = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(0)],
            )[0]  # (H, W) float32

            H, W = roi.shape[:2]
            px_per_mm = get_px_per_mm(config)
            boundary = detect_thread_boundary(roi)
            center_row = H - 1 - boundary
            ideal = np.flipud(generate_ideal_mask(px_per_mm, (H, W), center_row=center_row))
            registered_ideal, _ = register_mask(roi, ideal, roi_mask)
            save_gradcam_a(_P(img_path), roi, gradcam_map, registered_ideal,
                           score, _P(vis_dir), tool_id)
        except Exception as exc:
            warnings.warn(f"GradCAM visualization failed for {_P(img_path).name}: {exc}")

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self._model.state_dict(), "config": self._config}, path)

    def load(self, path: Path) -> None:
        # weights_only=False required: checkpoint contains non-tensor config dict.
        # Safe here because checkpoints are always self-generated by this pipeline.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._config = checkpoint["config"]
        self._transform = self._build_transform(
            self._config.get("image_size", [384, 512])
        )
        self._model = MILClassifier(dropout=self._config.get("dropout", 0.5))
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
