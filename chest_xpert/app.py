"""Chest Xpert — Streamlit UI (integrated staged pipeline).

Stage 0: Image & modality validation
Stage A: Pretrained disease classifier
Stage B: LLM #1 prediction verification
Stage C: LLM #2 report generation (template-based fallback)
Stage D: Human review and export

No training or fine-tuning runs here. LLM stages are unavailable unless
CHESTXPERT_LLM_API_URL / CHESTXPERT_LLM_API_TOKEN are set in the environment.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import streamlit as st

from chest_xpert.config import (
    ALLOWED_EXTENSIONS,
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_SAMPLE_IMAGE,
    DEVICE_CONFIG,
    DISCLAIMER,
    LABELS_18,
    LLM_CONFIG,
    LLM_MODEL_CONFIG,
    MODALITY_CONFIG,
    MODEL_CONFIG,
)
from chest_xpert.explainability import GradCAM
from chest_xpert.inference import InferenceEngine
from chest_xpert.label_mapping import (
    map_outputs_to_labels,
    scores_dict,
    supported_labels,
    unsupported_labels,
)
from chest_xpert.modality import ModalityValidator, ModalityDecision
from chest_xpert.preprocessing import preprocess, validate_image
from chest_xpert.report import (
    generate_html,
    generate_pdf,
    ReportData,
    ExamDetails,
)
from chest_xpert.verification import VerificationEngine
from chest_xpert.model_registry import registry
from chest_xpert.config import LORA_CONFIG

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _reset_stage_results() -> None:
    """Clear results from prior analyses so nothing leaks into a new upload."""
    st.session_state.modality_result = None
    st.session_state.inference_result = None
    st.session_state.preprocessed = None
    st.session_state.verification_result = None
    st.session_state.heatmap = None
    st.session_state.report_html = None
    st.session_state.report_data = ReportData()
    st.session_state.stage_latencies = {}

for key, default in {
    "engine": None,
    "engine_device": None,
    "upload_name": None,
    "upload_path": None,
    "modality_result": None,
    "inference_result": None,
    "preprocessed": None,
    "verification_result": None,
    "heatmap": None,
    "report_html": None,
    "report_data": ReportData(),
    "stage_latencies": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _get_engine(device: str) -> InferenceEngine:
    if (
        st.session_state.engine is None
        or st.session_state.engine_device != device
    ):
        engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=device)
        st.session_state.engine = engine
        st.session_state.engine_device = device
    return st.session_state.engine


def _get_modality_validator() -> ModalityValidator:
    if "modality_validator" not in st.session_state:
        st.session_state.modality_validator = ModalityValidator()
    return st.session_state.modality_validator


def _get_verification_engine() -> VerificationEngine:
    if "verification_engine" not in st.session_state:
        st.session_state.verification_engine = VerificationEngine()
    return st.session_state.verification_engine


def _get_llm_engine() -> VerificationEngine:
    key = (
        st.session_state.get("llm_backend"),
        st.session_state.get("llm_model"),
        st.session_state.get("llm_token"),
        st.session_state.get("llm_force_cpu"),
        st.session_state.get("llm_vllm_url"),
        st.session_state.get("llm_load_in_4bit"),
    )
    if (
        "llm_engine" not in st.session_state
        or st.session_state.get("llm_engine_key") != key
    ):
        from chest_xpert.verification import LLMModelClient
        client = LLMModelClient(
            model_id=st.session_state.get("llm_model") or LLM_MODEL_CONFIG.model_id,
            backend=st.session_state.get("llm_backend") or LLM_MODEL_CONFIG.backend,
            force_cpu=st.session_state.get("llm_force_cpu") or LLM_MODEL_CONFIG.force_cpu,
            vllm_url=st.session_state.get("llm_vllm_url") or LLM_MODEL_CONFIG.vllm_url,
            load_in_4bit=st.session_state.get("llm_load_in_4bit") or LLM_MODEL_CONFIG.load_in_4bit,
            lora_adapter_path=LORA_CONFIG.adapter_path if LORA_CONFIG.enabled else None,
        )
        if st.session_state.get("llm_token"):
            client._token = st.session_state.get("llm_token")
        st.session_state.llm_engine = VerificationEngine(client=client)
        st.session_state.llm_engine_key = key
    return st.session_state.llm_engine


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Upload & Pipeline (Stages 0–D)",
            "Explainability (Grad-CAM)",
            "Models & Methodology",
            "Model Comparison & Evaluation",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Model")
    st.write(f"**Selected:** {MODEL_CONFIG.name}")
    st.write(f"**Architecture:** {MODEL_CONFIG.architecture}")
    st.write(f"**Input:** {MODEL_CONFIG.input_resolution}×{MODEL_CONFIG.input_resolution} grayscale")
    auto_device = DEVICE_CONFIG.resolve()
    device = st.selectbox(
        "Device (CUDA preferred)",
        ["cuda", "cpu"],
        index=0 if auto_device == "cuda" else 1,
    )
    if st.button("Load / Reload Model"):
        try:
            eng = _get_engine(device)
            st.success(f"Model loaded on {eng.device}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load model: {exc}")
    st.divider()
    st.subheader("Stage 0 — Modality")
    st.write(f"**CLIP model:** {MODALITY_CONFIG.model_id}")
    st.write(f"**Qwen VLM:** {'configured' if LLM_MODEL_CONFIG.hf_token else 'not configured (HF token needed)'}")
    st.write(f"**Device:** {MODALITY_CONFIG.device}")
    st.caption(
        "Qwen VLM preferred (can see the image). CLIP + heuristic fallbacks. "
        "Not a validated clinical test."
    )
    st.divider()
    st.subheader("Stages B/C — LLM")
    st.write(f"**Generic LLM URL:** {'set' if LLM_CONFIG.api_url else 'not set'}")
    llm_backend = st.selectbox(
        "LLM backend",
        ["hf_api", "transformers_local", "vllm_server"],
        index=["hf_api", "transformers_local", "vllm_server"].index(LLM_MODEL_CONFIG.backend),
        key="llm_backend",
        help="hf_api = Hugging Face Inference API (free tier, needs token). "
             "transformers_local = load model locally via transformers. "
             "vllm_server = connect to self-hosted vLLM server.",
    )
    llm_model = st.text_input(
        "LLM model id",
        value=LLM_MODEL_CONFIG.model_id,
        key="llm_model",
        help="e.g. Qwen/Qwen3-VL-32B-Instruct (~16 GB VRAM in 4-bit, "
             "sees the actual X-ray image) · Qwen/Qwen2.5-1.5B-Instruct "
             "(~2 GB, text-only, CPU OK) · Qwen/Qwen2.5-0.5B-Instruct "
             "(~1 GB, text-only, CPU OK) · TinyLlama/TinyLlama-1.1B-"
             "Chat-v1.0 (~1 GB, text-only, CPU OK)",
    )
    hf_token = st.text_input(
        "HF access token (for hf_api)",
        type="password",
        value=LLM_MODEL_CONFIG.hf_token or "",
        key="llm_token",
        help="Set CHESTXPERT_HF_TOKEN in your shell to avoid typing each time.",
    )
    vllm_url = st.text_input(
        "vLLM server URL (for vllm_server)",
        value=LLM_MODEL_CONFIG.vllm_url or "",
        key="llm_vllm_url",
        help="e.g. http://localhost:8000 (run vLLM server separately)",
    )
    force_cpu = st.checkbox(
        "Force CPU loading (for local models)",
        value=LLM_MODEL_CONFIG.force_cpu,
        key="llm_force_cpu",
        help="Enable if your GPU is out of memory. Small models like "
             "Qwen2.5-1.5B (~2 GB) can run on CPU. Qwen3-VL-32B needs "
             "GPU VRAM (~16 GB in 4-bit).",
    )
    load_in_4bit = st.checkbox(
        "Use 4-bit quantization (for Qwen3-VL-32B)",
        value=LLM_MODEL_CONFIG.load_in_4bit,
        key="llm_load_in_4bit",
        help="Enable 4-bit quantization to fit Qwen3-VL-32B on 32 GB "
             "V100. Requires bitsandbytes.",
    )
    st.caption(
        "Recommended free models: Qwen/Qwen3-VL-32B-Instruct (~16 GB VRAM "
        "in 4-bit, can see the actual X-ray image) · Qwen/Qwen2.5-1.5B-"
        "Instruct (~2 GB, text-only, CPU OK) · Qwen/Qwen2.5-0.5B-Instruct "
        "(~1 GB, text-only, CPU OK) · TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
        "(~1 GB, text-only, CPU OK). Never share your token in chat; it is "
        "not saved in source code."
    )
    st.divider()
    st.markdown(
        f"<small>{DISCLAIMER}</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_score_card(label: str, mapping) -> None:
    if mapping.supported and mapping.score is not None:
        st.metric(label=label, value=f"{mapping.score:.4f}", delta=None)
        st.caption("Supported · model output")
    else:
        st.metric(label=label, value="—")
        st.caption("Unsupported · no model output")


# ---------------------------------------------------------------------------
# Pipeline page
# ---------------------------------------------------------------------------

def page_pipeline() -> None:
    st.header("Upload & Pipeline (Stages 0–D)")

    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
        help="PNG, JPG, JPEG or DICOM. Max 50 MB.",
    )
    use_sample = st.button("Use bundled sample image")
    sample_path: Optional[str] = None
    if use_sample and DEFAULT_SAMPLE_IMAGE.exists():
        sample_path = str(DEFAULT_SAMPLE_IMAGE)

    # New upload → invalidate all previous results.
    # Only reset when a genuinely new file is uploaded (compare by name).
    path_to_run: Optional[str] = st.session_state.get("upload_path")
    if uploaded is not None:
        if st.session_state.get("upload_name") != uploaded.name:
            _reset_stage_results()
        tmp_path = Path("/tmp/chest_xpert_upload") / uploaded.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getbuffer())
        path_to_run = str(tmp_path)
        st.session_state.upload_name = uploaded.name
        st.session_state.upload_path = path_to_run
    elif sample_path:
        if st.session_state.get("upload_name") != DEFAULT_SAMPLE_IMAGE.name:
            _reset_stage_results()
        path_to_run = sample_path
        st.session_state.upload_name = DEFAULT_SAMPLE_IMAGE.name
        st.session_state.upload_path = sample_path

    if not path_to_run:
        st.info("Upload a chest X-ray or use the bundled sample to begin the pipeline.")
        return

    validation = validate_image(path_to_run)

    # ---------- Stage 0: file validation ----------
    st.subheader("Stage 0 — Image & Modality Validation")
    if not validation.ok:
        st.error(f"File validation failed: {validation.reason}")
        st.session_state.modality_result = None
        return

    with st.container():
        c_img, c_meta = st.columns([2, 1])
        with c_img:
            st.image(
                path_to_run,
                caption=f"Uploaded: {st.session_state.upload_name}",
                use_container_width=True,
            )
        with c_meta:
            st.write("**File validation:** ✅ Passed")
            st.write(f"Size: {validation.size_mb} MB")
            st.write(f"Dimensions: {validation.dims}")
            st.write(f"Channels: {validation.channels}")

    # Modality check (CLIP zero-shot).
    validator = _get_modality_validator()
    with st.spinner("Checking image modality (chest X-ray vs. other)..."):
        modality_result = validator.validate(path_to_run)
    st.session_state.modality_result = modality_result

    # Render modality decision.
    if modality_result.is_technical_failure:
        st.error(
            "Modality check could not be completed (technical failure). "
            "This is not a judgment that the image is invalid. "
            f"Detail: {modality_result.reason}"
        )
        st.info("You may still proceed at your own risk, or upload a different image.")
        proceed = st.button("Proceed despite modality check failure", type="secondary")
        if not proceed:
            return
        st.session_state.report_data.modality_accepted = None
        st.session_state.report_data.modality_decision = "technical_failure"
        st.session_state.report_data.modality_reason = modality_result.reason
        st.session_state.report_data.modality_model = modality_result.model_id
        st.session_state.report_data.modality_ms = modality_result.latency_ms
    elif modality_result.is_rejected:
        st.error(
            "Input rejected: The uploaded image does not appear to be a chest X-ray. "
            "Please upload a valid chest X-ray image to continue."
        )
        st.session_state.report_data.modality_accepted = False
        st.session_state.report_data.modality_decision = "rejected"
        st.session_state.report_data.modality_reason = modality_result.reason
        st.session_state.report_data.modality_model = modality_result.model_id
        st.session_state.report_data.modality_ms = modality_result.latency_ms
        return
    elif modality_result.is_uncertain:
        st.warning(
            "Unable to verify the image type. Please upload a clear chest X-ray "
            "image and try again."
        )
        st.session_state.report_data.modality_accepted = None
        st.session_state.report_data.modality_decision = "uncertain"
        st.session_state.report_data.modality_reason = modality_result.reason
        st.session_state.report_data.modality_model = modality_result.model_id
        st.session_state.report_data.modality_ms = modality_result.latency_ms
        proceed = st.button(
            "Proceed anyway (modality unverified — not recommended)",
            type="secondary",
            key="proceed_uncertain",
        )
        if not proceed:
            return
        st.session_state.report_data.modality_accepted = None
        st.session_state.report_data.modality_decision = "uncertain"
    else:
        # accepted
        st.success("Modality check passed: image appears to be a chest X-ray.")
        st.session_state.report_data.modality_accepted = True
        st.session_state.report_data.modality_decision = "accepted"
        st.session_state.report_data.modality_reason = modality_result.reason
        st.session_state.report_data.modality_model = modality_result.model_id
        st.session_state.report_data.modality_ms = modality_result.latency_ms

    # Show modality evidence (transparent, not a diagnosis).
    with st.expander("Modality evidence (CLIP zero-shot, heuristic — not validated)"):
        if modality_result.all_probs:
            for label, p in sorted(
                modality_result.all_probs.items(), key=lambda kv: -kv[1]
            ):
                st.write(f"{label}: {p:.3f}")
        st.write(f"Chest X-ray prob: {modality_result.chest_xray_prob}")
        st.write(f"Best other: {modality_result.best_other_label} ({modality_result.best_other_prob})")
        st.caption(modality_result.reason)

    # ---------- Stage A: disease classifier ----------
    st.divider()
    st.subheader("Stage A — Pretrained Disease Classifier")
    run_a = st.button("Run disease classification", type="primary")
    if run_a or st.session_state.inference_result is not None:
        if run_a or st.session_state.inference_result is None:
            try:
                device_resolved = DEVICE_CONFIG.resolve()
                engine = _get_engine(device_resolved)
                pp = preprocess(path_to_run, device_resolved)
                result = engine.predict(pp.tensor)
                st.session_state.inference_result = result
                st.session_state.preprocessed = pp
                st.success("Classification complete. Backend logs have latency details.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Disease classification failed: {exc}")
                st.session_state.inference_result = None
                return

        result = st.session_state.inference_result
        mappings = map_outputs_to_labels(result.raw, result.model_name)
        st.session_state.report_data.model_name = result.model_name
        st.session_state.report_data.model_version = "pre-trained weights (torchxrayvision)"
        st.session_state.report_data.device = result.device
        st.session_state.report_data.mappings = mappings
        st.session_state.report_data.unsupported_labels = result.unsupported_labels

        st.write("**Prediction scores — all 18 labels**")
        half = 9
        left = mappings[:half]
        right = mappings[half:]
        c_left, c_right = st.columns(2)
        with c_left:
            for m in left:
                _render_score_card(m.requested_label, m)
        with c_right:
            for m in right:
                _render_score_card(m.requested_label, m)

        if result.unsupported_labels:
            st.warning("Unsupported labels: " + ", ".join(result.unsupported_labels))
        else:
            st.caption("All 18 requested labels supported by the selected model.")

    # ---------- Stage B: LLM #1 verification ----------
    st.divider()
    st.subheader("Stage B — Prediction Verification (LLM #1)")
    if st.session_state.inference_result is None:
        st.info("Run Stage A classification to enable verification.")
    else:
        result = st.session_state.inference_result
        mappings = map_outputs_to_labels(result.raw, result.model_name)
        model_doc = (
            "torchxrayvision DenseNet-121 (densenet121-res224-all), trained by "
            "other researchers on mixed public chest X-ray datasets. Scores are raw "
            "model outputs, not calibrated probabilities."
        )
        run_b = st.button("Run LLM #1 verification", type="primary")
        if run_b or st.session_state.verification_result is not None:
            if run_b or st.session_state.verification_result is None:
                engine_ver = _get_llm_engine()
                vr = engine_ver.verify(
                    mappings=mappings,
                    model_doc=model_doc,
                    image_path=path_to_run,
                )
                st.session_state.verification_result = vr
                rd = st.session_state.report_data
                rd.verification_result = vr
                rd.verification_available = vr.available
                rd.verification_error = vr.error
                rd.image_quality_note = vr.image_assessment
                st.session_state.report_data = rd

            vr = st.session_state.verification_result
            if not vr.available:
                st.info(
                    "LLM #1 verification unavailable: "
                    + (vr.error or "no LLM endpoint configured.")
                )
                st.caption(
                    "Report will proceed using the deterministic template fallback "
                    "(verification marked as not completed)."
                )
            else:
                st.success("LLM #1 verification complete.")
                if vr.image_assessment:
                    st.write(f"**Image assessment:** {vr.image_assessment}")
                n_cons = sum(1 for it in vr.items if it.assessment == "consistent")
                n_incon = sum(1 for it in vr.items if it.assessment == "inconsistent")
                n_unc = sum(1 for it in vr.items if it.assessment == "uncertain")
                c1, c2, c3 = st.columns(3)
                c1.metric("Consistent", str(n_cons))
                c2.metric("Inconsistent", str(n_incon))
                c3.metric("Uncertain", str(n_unc))
                for it in vr.items:
                    with st.expander(f"{it.pathology} — {it.assessment}"):
                        st.write(f"Classifier score: {it.classifier_score}")
                        st.write(f"Assessment: {it.assessment}")
                        st.write(f"Rationale: {it.rationale}")
                        st.write(f"Uncertainty: {it.uncertainty}")
                        st.write(f"Human review recommended: {it.human_review_recommended}")
                if vr.disagreements():
                    st.warning(
                        "Classifier–LLM disagreements detected. Both outputs are "
                        "preserved; neither is treated as ground truth."
                    )
            st.caption(
                "LLM verification is a secondary model-generated assessment, "
                "not ground truth or a substitute for a radiologist."
            )

    # ---------- Stages C/D: report ----------
    st.divider()
    st.subheader("Stages C & D — Report Generation & Review")
    rd = st.session_state.report_data

    st.write("**Examination details**")
    c1, c2 = st.columns(2)
    with c1:
        rd.exam.patient_id = st.text_input("Patient ID", value=rd.exam.patient_id or "")
        rd.exam.exam_date = st.text_input("Exam date", value=rd.exam.exam_date or "")
        rd.exam.modality = st.text_input("Modality", value=rd.exam.modality or "Chest X-ray")
        rd.exam.view = st.text_input("View", value=rd.exam.view or "")
    with c2:
        rd.exam.institution = st.text_input("Institution", value=rd.exam.institution or "")
        rd.exam.referring_physician = st.text_input("Referring physician", value=rd.exam.referring_physician or "")
        rd.device = st.text_input("Device", value=rd.device or "")
        rd.generated_at = st.text_input("Generated (UTC)", value=rd.generated_at or "")

    st.write("**Impression (draft)**")
    rd.impression = st.text_area("Impression", value=rd.impression or "", height=120)
    st.write("**Reviewer**")
    c3, c4 = st.columns(2)
    with c3:
        rd.reviewer_name = st.text_input("Reviewer name", value=rd.reviewer_name or "")
    with c4:
        rd.review_status = st.selectbox(
            "Review status",
            ["Pending", "Reviewed", "Corrected", "Signed off"],
            index=["Pending", "Reviewed", "Corrected", "Signed off"].index(
                rd.review_status or "Pending"
            ),
        )
    rd.notes = st.text_area("Notes", value=rd.notes or "", height=80)
    st.session_state.report_data = rd

    if st.button("Generate report", type="primary"):
        # Report findings are already in verification_result.report_findings
        # from the single LLM call in Stage B.
        html = generate_html(rd)
        st.session_state.report_html = html
        st.success("Report generated.")

    if "report_html" in st.session_state and st.session_state.report_html:
        st.subheader("Report preview")
        st.components.v1.html(st.session_state.report_html, height=700, scrolling=True)

        cpdf, cprint = st.columns(2)
        with cpdf:
            if st.button("Download PDF"):
                import tempfile
                out = Path(tempfile.mkdtemp()) / "chest_xpert_report.pdf"
                generate_pdf(rd, out)
                with open(out, "rb") as f:
                    st.download_button(
                        "Save PDF",
                        data=f.read(),
                        file_name="chest_xpert_report.pdf",
                        mime="application/pdf",
                    )
        with cprint:
            st.markdown(
                '<button onclick="window.print();" style="padding:8px 14px;">Print (browser)</button>',
                unsafe_allow_html=True,
            )
        st.info(
            "Every report is an AI-generated research draft requiring review by "
            "a qualified radiologist. Not a final clinical diagnosis."
        )


# ---------------------------------------------------------------------------
# Other pages (preserved)
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    st.header("Dashboard")
    st.write("Chest Xpert — chest X-ray multi-label findings using pre-trained models.")
    info = registry()[MODEL_CONFIG.name]
    n_supported = sum(1 for ls in info.supported_labels if ls.supported)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requested labels", "18")
    c2.metric("Supported by model", str(n_supported))
    c3.metric("Unsupported", str(18 - n_supported))
    c4.metric("Device", DEVICE_CONFIG.resolve())
    st.divider()
    st.subheader("Pipeline")
    st.write(
        "1. **Upload & Pipeline:** upload an image → modality validation → disease "
        "classification → LLM verification → report.\n"
        "2. **Explainability:** Grad-CAM for a supported finding.\n"
    )
    st.info(
        "Scores are raw model outputs, not calibrated probabilities or clinical "
        "confidence. Every report is an AI-generated research draft requiring "
        "review by a qualified radiologist."
    )


def page_explainability() -> None:
    st.header("Explainability (Grad-CAM)")
    if st.session_state.inference_result is None or st.session_state.preprocessed is None:
        st.info("Run disease classification in the pipeline page first.")
        return

    result = st.session_state.inference_result
    pp = st.session_state.preprocessed
    mappings = map_outputs_to_labels(result.raw, result.model_name)
    supportable = [m for m in mappings if m.supported]
    if not supportable:
        st.warning("No supported labels available for Grad-CAM.")
        return

    st.write("Select a supported finding to visualize. Highlighted regions are "
             "model explanations, not proof that a disease is present.")
    target_label = st.selectbox("Finding", [m.requested_label for m in supportable])

    if st.button("Generate Grad-CAM", type="primary"):
        try:
            engine = _get_engine(result.device)
            cam = GradCAM(engine.model, target_layer_name="features.norm5")
            import numpy as np
            from PIL import Image
            tensor = pp.tensor
            disp = tensor.detach().cpu().numpy()[0, 0]
            disp = np.clip((disp + 1024) / 2048, 0, 1)
            grad = cam.generate(
                input_tensor=tensor,
                target_label=target_label,
                model_outputs=result.model_outputs,
                original_image=disp,
            )
            st.session_state.heatmap = grad
            st.success(f"Grad-CAM generated for '{target_label}' "
                       f"(target layer: {grad.target_layer}).")
            cam.close()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Grad-CAM failed: {exc}")

    if st.session_state.heatmap is not None:
        grad = st.session_state.heatmap
        st.subheader(f"Grad-CAM — {grad.target_label}")
        c1, c2 = st.columns(2)
        with c1:
            st.image(grad.overlay, caption="Heatmap overlaid on original", use_container_width=True)
        with c2:
            st.image(grad.heatmap, caption="Raw heatmap", use_container_width=True, clamp=True)
        st.caption(
            f"Target layer: {grad.target_layer} · Class index: {grad.class_index} · "
            "Explanation, not a diagnosis."
        )
        import tempfile
        from PIL import Image as PILImage
        tmp = Path(tempfile.mkdtemp()) / "gradcam.png"
        PILImage.fromarray(grad.overlay).save(tmp)
        st.session_state.report_data.heatmap_path = str(tmp)
        st.session_state.report_data.heatmap_label = grad.target_label


def page_models_methodology() -> None:
    st.header("Models & Methodology")
    st.write("Documentation of the selected pre-trained model and its training.")
    info = registry()[MODEL_CONFIG.name]
    fields = [
        ("Model name", info.name),
        ("Architecture", info.architecture),
        ("Parameter count", info.parameter_count),
        ("Authors", info.authors),
        ("Organization", info.organization),
        ("Publication", info.publication),
        ("Repository", info.repository),
        ("Training datasets", info.training_datasets),
        ("Training objective", info.training_objective),
        ("Loss function", info.loss_function),
        ("Input dimensions", info.input_dimensions),
        ("Preprocessing", info.preprocessing),
        ("Multi-label head", info.multi_label_head),
        ("License", info.license),
        ("Permitted uses", info.permitted_uses),
        ("Limitations", info.limitations),
        ("Evaluation datasets (original)", info.evaluation_datasets),
        ("Published metrics", info.published_metrics),
    ]
    for label, value in fields:
        st.markdown(f"**{label}:** {value}")

    st.divider()
    st.subheader("Stage 0 — Modality model")
    st.write(f"Primary: CLIP ({MODALITY_CONFIG.model_id})")
    st.write(f"Device: CPU")
    st.caption("Heuristic gate — not a validated clinical test.")

    st.divider()
    st.subheader("Stages B/C — LLM")
    st.write(f"Generic LLM URL: {'set' if LLM_CONFIG.api_url else 'not set'}")
    st.write(f"LLM backend: {LLM_MODEL_CONFIG.backend}")
    st.write(f"LLM model: {LLM_MODEL_CONFIG.model_id}")
    st.write(f"HF Token: {'configured' if LLM_MODEL_CONFIG.hf_token else 'not configured'}")
    st.caption("Credentials from environment variables only; never hardcoded.")

    st.divider()
    st.subheader("Label support (18 requested)")
    sup = supported_labels(MODEL_CONFIG.name)
    unsup = unsupported_labels(MODEL_CONFIG.name)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Supported:", f"{len(sup)}/18")
        st.write(", ".join(sup))
    with c2:
        st.write("Unsupported:", f"{len(unsup)}/18")
        st.write(", ".join(unsup) if unsup else "None")

    st.divider()
    st.subheader("Attribution boundary")
    st.markdown(
        "- **Original model training:** torchxrayvision / CheXpert researchers. "
        "This project did **not** train or fine-tune the model.\n"
        "- **Chest Xpert implementation:** this project.\n"
        "- **Modality validation:** CLIP (OpenAI), zero-shot heuristic.\n"
        "- **LLM verification/report:** optional remote service; unavailable unless configured.\n"
        "- **Evaluation:** not yet performed."
    )


def page_evaluation() -> None:
    st.header("Model Comparison & Evaluation")
    st.write(
        "Evaluation computes per-label AUROC and AUPRC on a properly licensed, "
        "available test dataset. **Not yet run.** Results are not claimed until "
        "measured."
    )
    st.info(
        "No dataset has been evaluated by this project. Modality-validation, "
        "disease-classification, and LLM-verification metrics are not evaluated."
    )
    st.code(
        "from chest_xpert.evaluation import evaluate_dataset, save_report\n"
        "report = evaluate_dataset(image_paths, labels, patient_ids, dataset_name='my-test')\n"
        "save_report(report, 'evaluation_results.json')",
        language="python",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": page_dashboard,
    "Upload & Pipeline (Stages 0–D)": page_pipeline,
    "Explainability (Grad-CAM)": page_explainability,
    "Models & Methodology": page_models_methodology,
    "Model Comparison & Evaluation": page_evaluation,
}

PAGES[page]()

st.divider()
st.caption(
    f"{APP_TITLE} · {DISCLAIMER} "
    "Model: torchxrayvision DenseNet-121 (pre-trained weights by other researchers)."
)
