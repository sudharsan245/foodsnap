"""
Configuration module for AI-Generated Image Detection System.

This module contains all configurable parameters including:
- Model identifiers
- Decision thresholds
- Enum definitions for claim decisions
"""

from enum import Enum
from dataclasses import dataclass


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_ID = "Smogy/SMOGY-Ai-images-detector"
"""Hugging Face model identifier for the SMOGY AI image detector."""


# =============================================================================
# Decision Thresholds
# =============================================================================

@dataclass(frozen=True)
class DecisionThresholds:
    """
    Confidence thresholds for claim decision logic.
    
    These thresholds prioritize LOW FALSE POSITIVES to avoid wrongly 
    rejecting legitimate customer claims.
    
    Decision Logic:
        - AI confidence >= REJECT_THRESHOLD (0.85) → REJECT claim
        - AI confidence >= MANUAL_REVIEW_THRESHOLD (0.60) → MANUAL REVIEW
        - AI confidence < MANUAL_REVIEW_THRESHOLD → ACCEPT claim
    """
    REJECT_THRESHOLD: float = 0.85
    """AI probability >= this value triggers automatic claim rejection."""
    
    MANUAL_REVIEW_THRESHOLD: float = 0.60
    """AI probability >= this value (but < REJECT) triggers manual review."""


# Default thresholds instance
DEFAULT_THRESHOLDS = DecisionThresholds()


# =============================================================================
# Claim Decision Enum
# =============================================================================

class ClaimDecision(Enum):
    """
    Possible outcomes for a fraud detection claim decision.
    
    Attributes:
        ACCEPT: Claim appears legitimate, proceed with refund
        REJECT: High confidence of AI-generated image, block refund
        MANUAL_REVIEW: Uncertain, requires human verification
    """
    ACCEPT = "accept"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def emoji(self) -> str:
        """Return emoji representation for display."""
        return {
            ClaimDecision.ACCEPT: "🟢",
            ClaimDecision.REJECT: "🔴",
            ClaimDecision.MANUAL_REVIEW: "🟡"
        }[self]
    
    @property
    def description(self) -> str:
        """Return human-readable description."""
        return {
            ClaimDecision.ACCEPT: "Accept Claim - Image appears to be real",
            ClaimDecision.REJECT: "Reject Claim - High confidence AI-generated image",
            ClaimDecision.MANUAL_REVIEW: "Manual Review Required - Uncertain detection"
        }[self]
