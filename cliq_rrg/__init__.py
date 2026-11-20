"""CliQ-RRG: Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment
for QA-Style Radiology Report Generation.

Two-stage framework:
  Stage 1 - Disease-aware Visual-Textual Alignment (CNN encoder + PrAM)
  Stage 2 - GraphRAG-guided retrieval + LLM QA-style report generation
"""

from __future__ import annotations

from chest_xpert.config import LABELS_18

__all__ = ["LABELS_18"]

# Model artifact saved after Stage 1 training
CNN_MODEL_PATH = "cnn_model.pth"
