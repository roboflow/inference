"""
Visualization script for OCR gap analysis.
This helps understand the distribution of normalized gaps and where Otsu's threshold falls.
Now includes bimodality detection to match the v2 stitch_ocr_detections block.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_otsu_threshold(gaps: np.ndarray) -> tuple[float, bool, dict]:
    """Find natural break between intra-word and inter-word gaps using Otsu's method.

    Also detects whether the distribution is bimodal (two distinct groups) or
    unimodal (single group, suggesting single word or uniform spacing).

    Returns:
        Tuple of (threshold, is_bimodal, debug_info)
    """
    if len(gaps) < 2:
        return 0.0, False, {}

    hist, bin_edges = np.histogram(gaps, bins=min(50, len(gaps)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_thresh = 0.0
    best_variance = 0.0
    best_below_mean = 0.0
    best_above_mean = 0.0

    for t in bin_centers:
        below = gaps[gaps <= t]
        above = gaps[gaps > t]

        if len(below) == 0 or len(above) == 0:
            continue

        variance = len(below) * len(above) * (below.mean() - above.mean()) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t
            best_below_mean = below.mean()
            best_above_mean = above.mean()

    # Check if distribution is bimodal using several heuristics
    overall_std = gaps.std()
    overall_mean = gaps.mean()

    # Separation ratio: how far apart are the two class means relative to overall std
    mean_separation = abs(best_above_mean - best_below_mean)
    separation_ratio = mean_separation / overall_std if overall_std > 0 else 0

    # Bimodality criteria (matching v2.py) - MUST have meaningful word gaps:
    # The key insight is that real word gaps are typically 0.5+ in normalized units.
    # A distribution with all gaps < 0.3 is unimodal (single word), even if there
    # are outliers that inflate the mean separation.
    #
    # Primary criterion: above-class mean must indicate actual word gaps exist
    has_positive_word_gaps = best_above_mean > 0.3  # Word gaps should be clearly positive

    # Secondary criterion: good relative separation with positive gaps
    has_good_relative_separation = separation_ratio > 1.5 and mean_separation > 0.3

    # Must have positive word gaps to be considered bimodal
    is_bimodal = has_positive_word_gaps and (mean_separation > 0.3 or has_good_relative_separation)

    debug_info = {
        'separation_ratio': separation_ratio,
        'mean_separation': mean_separation,
        'overall_std': overall_std,
        'overall_mean': overall_mean,
        'best_below_mean': best_below_mean,
        'best_above_mean': best_above_mean,
        'best_variance': best_variance,
        'has_positive_word_gaps': has_positive_word_gaps,
        'has_good_relative_separation': has_good_relative_separation,
    }

    return best_thresh, is_bimodal, debug_info


def analyze_detections(detections_data: dict, reading_direction: str = "left_to_right"):
    """
    Analyze OCR detections and create visualizations.

    Args:
        detections_data: Dict with 'xyxy' (list of [x1,y1,x2,y2]) and 'class_name' (list of strings)
                        Can also include pre-computed values from debug output:
                        'is_bimodal', 'global_threshold', 'threshold_multiplier', 'all_normalized_gaps'
        reading_direction: Reading direction
    """
    xyxy = np.array(detections_data['xyxy'])
    class_names = detections_data['class_name']

    # Check if we have pre-computed values from the debug output
    has_precomputed = 'all_normalized_gaps' in detections_data

    if has_precomputed:
        # Use pre-computed values from the actual block
        all_gaps = np.array(detections_data['all_normalized_gaps'])
        threshold = detections_data.get('global_threshold', find_otsu_threshold(all_gaps)[0])
        is_bimodal = detections_data.get('is_bimodal', True)
        threshold_multiplier = detections_data.get('threshold_multiplier', 1.0)

        # Recompute to get debug info
        raw_threshold, _, debug_info = find_otsu_threshold(all_gaps)

        # Build gap_metadata from scratch for detailed analysis
        gap_metadata = []
        x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2
        y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        median_height = np.median(heights)
        line_tolerance = median_height * 0.5
        y_sorted_indices = np.argsort(y_centers)

        lines = []
        current_line = [y_sorted_indices[0]]
        current_line_y = y_centers[y_sorted_indices[0]]

        for idx in y_sorted_indices[1:]:
            if abs(y_centers[idx] - current_line_y) <= line_tolerance:
                current_line.append(idx)
                current_line_y = np.mean([y_centers[i] for i in current_line])
            else:
                lines.append(current_line)
                current_line = [idx]
                current_line_y = y_centers[idx]
        lines.append(current_line)

        gap_idx = 0
        for line_num, line in enumerate(lines):
            if len(line) < 2:
                continue
            line_x_centers = x_centers[line]
            line_widths = widths[line]
            x_sorted_order = np.argsort(line_x_centers)
            sorted_line = [line[i] for i in x_sorted_order]
            sorted_x_centers = line_x_centers[x_sorted_order]
            sorted_widths = line_widths[x_sorted_order]

            for i in range(1, len(sorted_line)):
                raw_gap = sorted_x_centers[i] - sorted_x_centers[i-1] - (sorted_widths[i-1] + sorted_widths[i]) / 2
                local_scale = (sorted_widths[i-1] + sorted_widths[i]) / 2
                normalized_gap = all_gaps[gap_idx] if gap_idx < len(all_gaps) else 0.0

                gap_metadata.append({
                    'line': line_num,
                    'pos': i,
                    'char_before': class_names[sorted_line[i-1]],
                    'char_after': class_names[sorted_line[i]],
                    'raw_gap': raw_gap,
                    'normalized_gap': normalized_gap,
                    'local_scale': local_scale
                })
                gap_idx += 1
    else:
        # Compute everything from scratch
        x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2
        y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        # Group into lines
        median_height = np.median(heights)
        line_tolerance = median_height * 0.5

        y_sorted_indices = np.argsort(y_centers)

        lines = []
        current_line = [y_sorted_indices[0]]
        current_line_y = y_centers[y_sorted_indices[0]]

        for idx in y_sorted_indices[1:]:
            if abs(y_centers[idx] - current_line_y) <= line_tolerance:
                current_line.append(idx)
                current_line_y = np.mean([y_centers[i] for i in current_line])
            else:
                lines.append(current_line)
                current_line = [idx]
                current_line_y = y_centers[idx]
        lines.append(current_line)

        # Collect all gaps with metadata
        all_gaps = []
        gap_metadata = []

        for line_num, line in enumerate(lines):
            if len(line) < 2:
                continue

            line_x_centers = x_centers[line]
            line_widths = widths[line]
            x_sorted_order = np.argsort(line_x_centers)

            sorted_line = [line[i] for i in x_sorted_order]
            sorted_x_centers = line_x_centers[x_sorted_order]
            sorted_widths = line_widths[x_sorted_order]

            for i in range(1, len(sorted_line)):
                raw_gap = sorted_x_centers[i] - sorted_x_centers[i-1] - (sorted_widths[i-1] + sorted_widths[i]) / 2
                local_scale = (sorted_widths[i-1] + sorted_widths[i]) / 2

                if local_scale > 0:
                    normalized_gap = raw_gap / local_scale
                else:
                    normalized_gap = 0.0

                all_gaps.append(normalized_gap)
                gap_metadata.append({
                    'line': line_num,
                    'pos': i,
                    'char_before': class_names[sorted_line[i-1]],
                    'char_after': class_names[sorted_line[i]],
                    'raw_gap': raw_gap,
                    'normalized_gap': normalized_gap,
                    'local_scale': local_scale
                })

        all_gaps = np.array(all_gaps)
        threshold, is_bimodal, debug_info = find_otsu_threshold(all_gaps)
        threshold_multiplier = 1.0

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Determine title color based on bimodality
    bimodal_status = "BIMODAL ✓" if is_bimodal else "UNIMODAL (single word mode)"
    bimodal_color = "green" if is_bimodal else "orange"

    # 1. Histogram of normalized gaps with Otsu threshold
    ax1 = axes[0, 0]
    ax1.hist(all_gaps, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Otsu threshold: {threshold:.3f}')
    ax1.set_xlabel('Normalized Gap (gap / avg_char_width)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Normalized Gaps\n[{bimodal_status}]', color=bimodal_color if not is_bimodal else 'black')
    ax1.legend()

    # Add bimodality metrics as text
    if debug_info:
        textstr = f"Sep Ratio: {debug_info['separation_ratio']:.3f}\nMean Sep: {debug_info['mean_separation']:.3f}\nAbove Mean: {debug_info['best_above_mean']:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    # 2. Sorted gaps plot to see the distribution shape
    ax2 = axes[0, 1]
    sorted_gaps = np.sort(all_gaps)
    ax2.plot(sorted_gaps, marker='.', markersize=3, linestyle='-', alpha=0.7)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Otsu threshold: {threshold:.3f}')
    ax2.set_xlabel('Gap Index (sorted)')
    ax2.set_ylabel('Normalized Gap')
    title2 = 'Sorted Gaps (looking for bimodal "elbow")'
    if not is_bimodal:
        title2 += '\n⚠️ No clear elbow detected - treating as single word'
    ax2.set_title(title2)
    ax2.legend()

    # 3. Gap values colored by classification
    ax3 = axes[1, 0]
    below_thresh = all_gaps[all_gaps <= threshold]
    above_thresh = all_gaps[all_gaps > threshold]

    ax3.hist(below_thresh, bins=30, alpha=0.7, label=f'Intra-word ({len(below_thresh)})', color='blue')
    ax3.hist(above_thresh, bins=30, alpha=0.7, label=f'Inter-word ({len(above_thresh)})', color='orange')
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Normalized Gap')
    ax3.set_ylabel('Frequency')

    if is_bimodal:
        ax3.set_title('Gaps Classified by Otsu Threshold')
    else:
        ax3.set_title('Gaps Classified by Otsu Threshold\n⚠️ Classification ignored (unimodal)', color='orange')
    ax3.legend()

    # 4. Box plot showing the two distributions OR bimodality decision explanation
    ax4 = axes[1, 1]

    if is_bimodal:
        data_to_plot = [below_thresh, above_thresh]
        bp = ax4.boxplot(data_to_plot, labels=['Intra-word\n(same word)', 'Inter-word\n(word break)'])
        ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_ylabel('Normalized Gap')
        ax4.set_title('Distribution Comparison')
    else:
        # Show bimodality decision breakdown
        ax4.axis('off')
        decision_text = "BIMODALITY DECISION\n" + "="*30 + "\n\n"

        if debug_info:
            sep_ratio = debug_info['separation_ratio']
            mean_sep = debug_info['mean_separation']
            above_mean = debug_info['best_above_mean']

            # Check each criterion (matching v2.py)
            has_positive_gaps = above_mean > 0.3
            has_good_relative = sep_ratio > 1.5 and mean_sep > 0.3

            decision_text += f"Above-class Mean: {above_mean:.3f}\n"
            decision_text += f"  Threshold: > 0.3  →  {'PASS ✓' if has_positive_gaps else 'FAIL ✗'}\n"
            decision_text += f"  (Must have real word gaps)\n\n"

            decision_text += f"Mean Separation: {mean_sep:.3f}\n"
            decision_text += f"  Threshold: > 0.3  →  {'PASS ✓' if mean_sep > 0.3 else 'FAIL ✗'}\n\n"

            decision_text += f"Separation Ratio: {sep_ratio:.3f}\n"
            decision_text += f"  Threshold: > 1.5 (with MeanSep>0.3)\n"
            decision_text += f"  →  {'PASS ✓' if has_good_relative else 'FAIL ✗'}\n\n"

            decision_text += "="*30 + "\n"
            decision_text += f"Formula:\n"
            decision_text += f"  AboveMean>0.3 AND\n"
            decision_text += f"  (MeanSep>0.3 OR GoodRelativeSep)\n\n"
            decision_text += f"Result: {has_positive_gaps} AND ({mean_sep > 0.3} OR {has_good_relative})\n"
            decision_text += f"        = {is_bimodal}\n\n"

            decision_text += "CONCLUSION: Treating all gaps as intra-word\n"
            decision_text += "(No word breaks will be inserted)"

        ax4.text(0.5, 0.5, decision_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('ocr_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: ocr_gap_analysis.png")

    # Create a detailed text report
    print("\n" + "="*60)
    print("GAP ANALYSIS REPORT")
    print("="*60)
    print(f"\nTotal gaps analyzed: {len(all_gaps)}")
    print(f"Otsu threshold: {threshold:.4f}")
    if has_precomputed:
        print(f"Threshold multiplier: {threshold_multiplier}")

    # Bimodality section
    print("\n" + "-"*60)
    print("BIMODALITY DETECTION")
    print("-"*60)
    print(f"  Is Bimodal: {is_bimodal}")
    if debug_info:
        print(f"  Separation Ratio: {debug_info['separation_ratio']:.4f}")
        print(f"  Mean Separation: {debug_info['mean_separation']:.4f} (threshold: >0.3)")
        print(f"  Below-class mean: {debug_info['best_below_mean']:.4f}")
        print(f"  Above-class mean: {debug_info['best_above_mean']:.4f} (threshold: >0.2 for word gaps)")
        print(f"  Criteria:")
        print(f"    has_positive_word_gaps (above_mean > 0.3): {debug_info.get('has_positive_word_gaps', 'N/A')}")
        print(f"    has_good_relative_separation: {debug_info.get('has_good_relative_separation', 'N/A')}")

    if not is_bimodal:
        print("\n  ⚠️  Distribution is UNIMODAL - all characters will be treated as")
        print("     a single word (no word breaks inserted)")

    print(f"\nGaps below threshold (intra-word): {len(below_thresh)} ({100*len(below_thresh)/len(all_gaps):.1f}%)")
    print(f"Gaps above threshold (inter-word): {len(above_thresh)} ({100*len(above_thresh)/len(all_gaps):.1f}%)")

    print(f"\nGap statistics:")
    print(f"  Min: {all_gaps.min():.4f}")
    print(f"  Max: {all_gaps.max():.4f}")
    print(f"  Mean: {all_gaps.mean():.4f}")
    print(f"  Median: {np.median(all_gaps):.4f}")
    print(f"  Std: {all_gaps.std():.4f}")

    if is_bimodal:
        # Show misclassified examples (gaps near the threshold)
        print("\n" + "-"*60)
        print("GAPS NEAR THRESHOLD (potential misclassifications)")
        print("-"*60)

        # Sort by distance from threshold
        distances = np.abs(all_gaps - threshold)
        near_threshold_indices = np.argsort(distances)[:20]

        for idx in near_threshold_indices:
            if idx < len(gap_metadata):
                meta = gap_metadata[idx]
                classification = "WORD_BREAK" if meta['normalized_gap'] > threshold else "SAME_WORD"
                distance = meta['normalized_gap'] - threshold
                print(f"  '{meta['char_before']}' -> '{meta['char_after']}': "
                      f"gap={meta['normalized_gap']:.4f} ({classification}, dist={distance:+.4f})")

        # Show clear word breaks (large gaps)
        print("\n" + "-"*60)
        print("CLEAR WORD BREAKS (largest gaps)")
        print("-"*60)

        largest_gap_indices = np.argsort(all_gaps)[-10:][::-1]
        for idx in largest_gap_indices:
            if idx < len(gap_metadata):
                meta = gap_metadata[idx]
                print(f"  '{meta['char_before']}' -> '{meta['char_after']}': gap={meta['normalized_gap']:.4f}")

        # Show clear same-word (smallest gaps)
        print("\n" + "-"*60)
        print("CLEAR SAME-WORD (smallest gaps)")
        print("-"*60)

        smallest_gap_indices = np.argsort(all_gaps)[:10]
        for idx in smallest_gap_indices:
            if idx < len(gap_metadata):
                meta = gap_metadata[idx]
                print(f"  '{meta['char_before']}' -> '{meta['char_after']}': gap={meta['normalized_gap']:.4f}")
    else:
        print("\n" + "-"*60)
        print("ALL GAPS (unimodal - no word breaks)")
        print("-"*60)
        for idx, meta in enumerate(gap_metadata[:20]):
            print(f"  '{meta['char_before']}' -> '{meta['char_after']}': gap={meta['normalized_gap']:.4f}")
        if len(gap_metadata) > 20:
            print(f"  ... and {len(gap_metadata) - 20} more gaps")

    # Save detailed gap data to JSON
    gap_report = {
        'threshold': float(threshold),
        'is_bimodal': bool(is_bimodal),
        'bimodality_metrics': {
            'separation_ratio': float(debug_info['separation_ratio']) if debug_info else None,
            'mean_separation': float(debug_info['mean_separation']) if debug_info else None,
            'below_class_mean': float(debug_info['best_below_mean']) if debug_info else None,
            'above_class_mean': float(debug_info['best_above_mean']) if debug_info else None,
            'has_positive_word_gaps': bool(debug_info.get('has_positive_word_gaps')) if debug_info else None,
            'has_good_relative_separation': bool(debug_info.get('has_good_relative_separation')) if debug_info else None,
        },
        'total_gaps': len(all_gaps),
        'intra_word_count': len(below_thresh),
        'inter_word_count': len(above_thresh),
        'statistics': {
            'min': float(all_gaps.min()),
            'max': float(all_gaps.max()),
            'mean': float(all_gaps.mean()),
            'median': float(np.median(all_gaps)),
            'std': float(all_gaps.std())
        },
        'all_gaps': [
            {
                'char_before': m['char_before'],
                'char_after': m['char_after'],
                'normalized_gap': float(m['normalized_gap']),
                'raw_gap': float(m['raw_gap']),
                'classification': 'inter_word' if m['normalized_gap'] > threshold else 'intra_word',
                'effective_classification': 'intra_word' if not is_bimodal else ('inter_word' if m['normalized_gap'] > threshold else 'intra_word')
            }
            for m in gap_metadata
        ]
    }

    with open('ocr_gap_report.json', 'w') as f:
        json.dump(gap_report, f, indent=2)

    print(f"\nSaved: ocr_gap_report.json")

    return threshold, is_bimodal, all_gaps, gap_metadata


def analyze_detections_to_folder(detections_data: dict, output_dir: Path, debug_id: str, reading_direction: str = "left_to_right"):
    """
    Analyze OCR detections and save outputs to a specific folder.

    Args:
        detections_data: Detection data dict
        output_dir: Directory to save outputs
        debug_id: Unique identifier for this analysis
        reading_direction: Reading direction
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Temporarily change to output directory for file saves
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        result = analyze_detections(detections_data, reading_direction)
    finally:
        os.chdir(original_cwd)

    return result


# Example usage - you'll need to provide actual detection data
if __name__ == "__main__":
    import sys
    import os

    # Check for debug files in the new folder structure
    debug_dir = Path('ocr_debug')
    output_base_dir = Path('ocr_reports')

    if debug_dir.exists() and list(debug_dir.glob('ocr_detections_*.json')):
        # Process all debug files in the folder
        debug_files = sorted(debug_dir.glob('ocr_detections_*.json'), key=lambda p: p.stat().st_mtime)
        print(f"Found {len(debug_files)} debug file(s) in {debug_dir}")
        print(f"Output will be saved to {output_base_dir}/<debug_id>/")
        print("-" * 60)

        for debug_file in debug_files:
            with open(debug_file, 'r') as f:
                detections_data = json.load(f)

            debug_id = detections_data.get('debug_id', debug_file.stem.replace('ocr_detections_', ''))
            output_dir = output_base_dir / debug_id

            print(f"\n{'='*60}")
            print(f"Processing: {debug_file.name}")
            print(f"Output dir: {output_dir}")
            print(f"{'='*60}")

            analyze_detections_to_folder(detections_data, output_dir, debug_id)

        print(f"\n{'='*60}")
        print(f"Processed {len(debug_files)} debug files")
        print(f"Reports saved to: {output_base_dir}/")

    else:
        print("No debug files found in /data/ocr_debug/")
        print("\nRun the OCR workflow first to generate debug files.")
        print("\nExpected structure:")
        print("  /data/ocr_debug/ocr_detections_<uuid>.json")
        print("\nOutput will be saved to:")
        print("  /data/ocr_reports/<uuid>/ocr_gap_analysis.png")
        print("  /data/ocr_reports/<uuid>/ocr_gap_report.json")
