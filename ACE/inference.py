"""
Run inference on full audio files using a trained ConformerDecomposedModel.
The audio is processed in 20-second chunks, and the predictions are merged.
The final output is saved as a .lab file with chord annotations.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

from ACE.mir_evaluation import convert_predictions_decomposed, remove_short_chords
from ACE.models.conformer_decomposed import ConformerDecomposedModel
from ACE.preprocess.audio_processor import AudioChunkProcessor
from ACE.preprocess.transforms import CQTransform


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConformerDecomposedModel.load_from_checkpoint(
        checkpoint_path,
        vocabularies={"root": 13, "bass": 13, "onehot": 12},
        map_location=device,
        loss="consonance_decomposed",
    )
    model.eval().to(device)
    print(f"✅ Loaded model from {checkpoint_path}")
    return model


@torch.no_grad()
def predict(model: ConformerDecomposedModel, features: torch.Tensor):
    """Run inference on a single feature tensor."""
    device = next(model.parameters()).device
    features = features.to(device)
    outputs = model(features)
    root = outputs["root"].argmax(dim=-1).squeeze().cpu().numpy()
    bass = outputs["bass"].argmax(dim=-1).squeeze().cpu().numpy()
    chord = torch.sigmoid(outputs["onehot"]).squeeze().cpu().numpy()
    return root, bass, chord


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


def run_inference(audio_path: Path, checkpoint: Path, out_lab: Path):
    """Run inference on the entire audio by concatenating 20s predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(checkpoint))

    # Parameters
    sample_rate = 22050
    hop_length = 512
    chunk_dur = 20.0  # seconds, same as training

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
    total_dur = librosa.get_duration(filename=str(audio_path))
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

        root, bass, chord = predict(model, features)

        # Decode to intervals/labels for this chunk
        intervals, labels = convert_predictions_decomposed(
            root_predictions=root,
            bass_predictions=bass,
            chord_predictions=chord,
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
    args = parser.parse_args()

    run_inference(args.audio, args.ckpt, args.out)
