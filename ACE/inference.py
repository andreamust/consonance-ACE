"""
Run inference on full audio files using a trained ConformerDecomposedModel.
The audio is processed in 20-second chunks, and the predictions are merged.
The final output is saved as a .lab file with chord annotations.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

from ACE.mir_evaluation import convert_predictions_decomposed, remove_short_chords, convert_predictions
from ACE.models.conformer_decomposed import ConformerDecomposedModel
from ACE.preprocess.audio_processor import AudioChunkProcessor
from ACE.preprocess.transforms import CQTransform
from ACE.models.conformer import ConformerModel


def load_model(checkpoint_path: str, vocab_path: str | Path = "./ACE/chords_vocab.joblib", model_name: str = "conformer_decomposed"):
    """Load trained model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (model_name == "conformer"):
        model = ConformerModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            # loss="consonance_decomposed",
            vocab_path=vocab_path,
            # strict=False,
        )
        model.eval().to(device)
        print(f"✅ Loaded model from {checkpoint_path} and vocab from {vocab_path}")
        return model
    elif (model_name == "conformer_decomposed"):
        model = ConformerDecomposedModel.load_from_checkpoint(
            checkpoint_path,
            vocabularies={"root": 13, "bass": 13, "onehot": 12},
            map_location=device,
            loss="consonance_decomposed",
            vocab_path=vocab_path,
            strict=False,
        )
        model.eval().to(device)
        print(f"✅ Loaded model from {checkpoint_path} and vocab from {vocab_path}")
        return model


@torch.no_grad()
def predict(model, features):
    """Run inference on a single feature tensor.
    Returns:
      - dict(type='decomposed', root=..., bass=..., chord=...) for decomposed model
      - dict(type='conformer', preds=...) for plain conformer model (class indices)
    """
    device = next(model.parameters()).device
    features = features.to(device)
    outputs = model(features)

    # Decomposed model returns a dict with keys "root","bass","onehot"
    if isinstance(outputs, dict):
        root = outputs["root"].argmax(dim=-1).squeeze().cpu().numpy()
        bass = outputs["bass"].argmax(dim=-1).squeeze().cpu().numpy()
        chord = torch.sigmoid(outputs["onehot"]).squeeze().cpu().numpy()
        return {"type": "decomposed", "root": root, "bass": bass, "chord": chord}

    # Conformer model returns a logits tensor [B, T, C]
    if torch.is_tensor(outputs):
        preds = outputs.argmax(dim=-1).squeeze().cpu().numpy()  # (T,) for single batch
        return {"type": "conformer", "preds": preds}

    raise RuntimeError("Unknown model output type in predict()")


def write_lab(path: Path, intervals: np.ndarray, labels: list[str]):
    """Write a .lab file."""
    with open(path, "w", encoding="utf-8") as f:
        for (s, e), lab in zip(intervals, labels):
            f.write(f"{s:.6f}\t{e:.6f}\t{lab}\n")
    print(f"💾 Saved {path}")


def merge_identical_consecutive(intervals: np.ndarray, labels: list[str]):
    """Merge consecutive intervals with identical labels."""
    if len(labels) == 0:
        return intervals, labels

    merged_intervals = [intervals[0].tolist()]
    merged_labels = [labels[0]]

    for i in range(1, len(labels)):
        if labels[i] == merged_labels[-1]:
            # Extend previous interval end time
            merged_intervals[-1][1] = intervals[i][1]
        else:
            merged_intervals.append(intervals[i].tolist())
            merged_labels.append(labels[i])

    return np.array(merged_intervals), merged_labels


def run_inference(
    audio_path: Path, 
    checkpoint: Path,
    vocab_path: str | Path, 
    out_lab: Path, 
    chord_min_duration: float = 0.5,
    model_name: str = "conformer_decomposed",
    threshold: float = 0.5,
    chunk_dur: float = 20.0,
):
    """Run inference on the entire audio by concatenating 20s predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(checkpoint), vocab_path=vocab_path, model_name=model_name)

    # Parameters
    sample_rate = 22050
    hop_length = 512
    chunk_dur = chunk_dur # seconds, default is 20 same as training

    # Preprocessor that keeps audio in memory
    transform = CQTransform(sample_rate, hop_length)
    chunker = AudioChunkProcessor(
        audio_path=audio_path,
        target_sample_rate=sample_rate,
        hop_length=hop_length,
        max_sequence_length=chunk_dur,
        device=device,
        transform=transform,
        normalize=True,
    )

    # Get total duration
    total_dur = librosa.get_duration(path=str(audio_path))
    n_chunks = int(np.ceil(total_dur / chunk_dur))
    print(f"🔍 Processing {n_chunks} chunks of ~{chunk_dur:.1f}s each")

    all_intervals = []
    all_labels = []

    for i in range(n_chunks):
        onset = i * chunk_dur
        print(f"Chunk {i + 1}/{n_chunks} (start {onset:.1f}s)")
        features = chunker.process_chunk(onset=onset)

        # Ensure shape [1, 1, F, T]
        if features.ndim == 2:
            features = features.unsqueeze(0).unsqueeze(0)
        elif features.ndim == 3:
            features = features.unsqueeze(0)

        pred_res = predict(model, features)

        if pred_res["type"] == "decomposed":
            root = pred_res["root"]
            bass = pred_res["bass"]
            chord = pred_res["chord"]

            # Decode to intervals/labels for this chunk (decomposed)
            intervals, labels = convert_predictions_decomposed(
                root_predictions=root,
                bass_predictions=bass,
                chord_predictions=chord,
                segment_duration=chunk_dur,
                threshold=threshold,
                remove_short_min_duration=chord_min_duration,
            )
        else:
            # Conformer (single-class) predictions: preds is an array of class indices
            preds = pred_res["preds"]
            intervals, labels = convert_predictions(
                predictions=preds,
                vocabulary="complete",
                segment_duration=chunk_dur,
            )

        # Shift time by onset to place in global timeline
        if len(intervals) > 0:
            intervals = intervals.copy()
            intervals[:, 0] += onset
            intervals[:, 1] += onset
            all_intervals.append(intervals)
            all_labels.extend(labels)

    # Concatenate and merge
    if all_intervals:
        all_intervals = np.vstack(all_intervals)
        # First, remove short chords
        all_intervals, all_labels = remove_short_chords(all_intervals, all_labels)
        # Then, merge identical consecutive chords
        all_intervals, all_labels = merge_identical_consecutive(
            all_intervals, all_labels
        )

        out_lab.parent.mkdir(parents=True, exist_ok=True)
        write_lab(out_lab, all_intervals, all_labels)
    else:
        print("⚠️ No predictions produced.")


if __name__ == "__main__":
    # Example usage
    # python -m ACE.inference --audio path/to/audio.wav --out path/to/output.lab

    import argparse

    parser = argparse.ArgumentParser(description="Run full-audio inference.")
    parser.add_argument("--audio", type=Path, required=True, help="Input audio file")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("ACE/checkpoints/conformer_decomposed_smooth.ckpt"),
        help="Path to checkpoint file",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .lab file path")
    parser.add_argument(
        "--chord-min-duration",
        type=float,
        default=0.5,
        help="Minimum duration for chord segments (in seconds)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./ACE/chords_vocab.joblib",
        help="Optional path to chord vocabulary (joblib) to override default",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="conformer_decomposed",
        help="Model name: 'conformer' or 'conformer_decomposed'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for chord component activation (only for decomposed model)",
    )
    parser.add_argument(
        "--chunk-dur",
        type=float,
        default=20.0,
        help="Duration of audio chunks to proces s (in seconds)",
    )
    args = parser.parse_args()

    run_inference(
        args.audio, 
        args.ckpt, 
        args.vocab_path, 
        args.out, 
        args.chord_min_duration, 
        args.vocab_path, 
        args.model_name, 
        args.threshold, 
        args.chunk_dur
    )
