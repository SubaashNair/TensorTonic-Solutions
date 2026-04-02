import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    results = []
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_w = x2 - x1
        roi_h = y2 - y1
        output = []
        for i in range(output_size):
            row = []
            for j in range(output_size):
                hstart = y1 + int(math.floor(i * roi_h / output_size))
                hend = y1 + int(math.floor((i + 1) * roi_h / output_size))
                wstart = x1 + int(math.floor(j * roi_w / output_size))
                wend = x1 + int(math.floor((j + 1) * roi_w / output_size))
                if hend == hstart:
                    hend = hstart + 1
                if wend == wstart:
                    wend = wstart + 1
                max_val = max(
                    feature_map[r][c]
                    for r in range(hstart, hend)
                    for c in range(wstart, wend)
                )
                row.append(max_val)
            output.append(row)
        results.append(output)
    return results