import cv2
import csv
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from skimage.metrics import structural_similarity as ssim

BODY_PARTS = [
    'Nose', 'Microscope', 'Microscope_left', 'Microscope_right', 'Neck',
    'Ear_1', 'Ear_2', 'Back_1', 'Back_2', 'Back_3',
    'Top_tail', 'Mid_tail', 'Final_tail'
]

def detect_ROI(dlc_data, video_path, display_video=False, save_video=False):
    """
    Desc: Detects active exploration of two objects by a mouse across all video frames.
    Input: dlc_data (DataFrame) cleaned DLC tracking, video_path (str).
    Output: list [time_obj1, time_obj2, di_obj1, di_obj2, quadrant1, quadrant2].
    """

    body_part_positions = {bp: (dlc_data[f'{bp}_x'], dlc_data[f'{bp}_y'], dlc_data[f'{bp}_likelihood']) for bp in BODY_PARTS}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("active_exploring_detection.mp4", fourcc, fps, (frame_width, frame_height))

    object_positions, mean_radius, found_objects = select_first_frames(video_path)
    if not found_objects:
        return [-1, -1, -1, -1, -1, -1]

    quadrant1, quadrant2 = find_quadrant(object_positions, frame_width, frame_height)
    object1_center = (object_positions[0][0], object_positions[0][1])
    object2_center = (object_positions[1][0], object_positions[1][1])
    small_radius_1 = small_radius_2 = mean_radius
    large_radius_1 = int(small_radius_1 + 35)
    large_radius_2 = int(small_radius_2 + 35)

    time_active_exploration_obj1 = time_active_exploration_obj2 = 0
    bar_width, bar_height, label_offset = 400, 15, 280
    bar1_start, bar2_start = (400, 30), (400, 75)
    progress1 = progress2 = true_progress1 = true_progress2 = 0
    delay = int(1000 / (fps / 0.1))
    alignment_margin_degrees = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in range(len(dlc_data)):
        ret, frame = cap.read()
        if not ret:
            print("End of video or mismatch between DLC data and video length.")
            break

        nose_x, nose_y = body_part_positions['Nose'][0][frame_idx], body_part_positions['Nose'][1][frame_idx]
        neck_x, neck_y = body_part_positions['Neck'][0][frame_idx], body_part_positions['Neck'][1][frame_idx]
        ear1_x, ear1_y = body_part_positions['Ear_1'][0][frame_idx], body_part_positions['Ear_1'][1][frame_idx]
        ear2_x, ear2_y = body_part_positions['Ear_2'][0][frame_idx], body_part_positions['Ear_2'][1][frame_idx]
        center_ears_x, center_ears_y = (ear1_x + ear2_x) / 2, (ear1_y + ear2_y) / 2

        nose = (nose_x, nose_y)
        neck = (neck_x, neck_y)
        center_ears = (center_ears_x, center_ears_y)

        active_exploration_obj1 = False
        active_exploration_obj2 = False

        for obj_center, small_radius, large_radius in [(object1_center, small_radius_1, large_radius_1),(object2_center, small_radius_2, large_radius_2)]:
            # Condition 1: nose in small circle, all key points in large circle
            in_small_circle = np.linalg.norm(np.array(obj_center) - np.array(nose)) <= small_radius
            all_points_in_large_circle = (
                np.linalg.norm(np.array(obj_center) - np.array(nose)) <= large_radius and
                np.linalg.norm(np.array(obj_center) - np.array(neck)) <= large_radius and
                np.linalg.norm(np.array(obj_center) - np.array(center_ears)) <= large_radius
            )
            is_touching = in_small_circle and all_points_in_large_circle
            # Condition 2: head in large circle and mouse facing the object
            is_facing_object = all_points_in_large_circle and is_aligned_via_regression(nose, center_ears, neck, obj_center, alignment_margin_degrees)
            # Condition 3: mouse is directly over the object
            center_head = ((nose_x + neck_x + ear1_x + ear2_x) / 4, (nose_y + neck_y + ear1_y + ear2_y) / 4)
            centroid = compute_mouse_centroid(frame_idx, body_part_positions)
            is_over_object = (np.linalg.norm(np.array(obj_center) - np.array(center_head)) <= (small_radius / 2)) or (np.linalg.norm(np.array(obj_center) - np.array(centroid)) <= (small_radius / 2))

            if (is_touching or is_facing_object) and not is_over_object:
                if obj_center == object1_center:
                    active_exploration_obj1 = True
                else:
                    active_exploration_obj2 = True

        if active_exploration_obj1:
            true_progress1 = true_progress1 + 0.28
            progress1 = round(true_progress1)
            time_active_exploration_obj1 += 1 / fps
        if active_exploration_obj2:
            true_progress2 = true_progress2 + 0.28
            progress2 = round(true_progress2)
            time_active_exploration_obj2 += 1 / fps

        # Draw object circles and progress bars
        for ctr, sr, lr in [(object1_center, small_radius_1, large_radius_1), (object2_center, small_radius_2, large_radius_2)]:
            cv2.circle(frame, ctr, sr, (150, 0, 0), 4)
            cv2.circle(frame, ctr, lr, (255, 100, 0), 4)
        for bs, prog, label in [(bar1_start, progress1, "Object 1"), (bar2_start, progress2, "Object 2")]:
            cv2.rectangle(frame, bs, (bs[0] + bar_width, bs[1] + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, bs, (bs[0] + prog, bs[1] + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, label, (bs[0] - label_offset + 180, bs[1] + bar_height - 5), font, 0.6, (255, 255, 255), 2)

        if display_video:
            cv2.imshow("Circular ROI Analysis", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        if save_video:
            out.write(frame)

    cap.release()
    cv2.destroyAllWindows()

    total_time = time_active_exploration_obj1 + time_active_exploration_obj2
    if total_time > 0:
        di_obj1_as_a = (time_active_exploration_obj1 - time_active_exploration_obj2) / total_time
        di_obj2_as_a = (time_active_exploration_obj2 - time_active_exploration_obj1) / total_time
    else:
        print("No exploration detected; Discrimination Index cannot be calculated.")
        di_obj1_as_a = di_obj2_as_a = 0

    return [time_active_exploration_obj1, time_active_exploration_obj2, di_obj1_as_a, di_obj2_as_a, quadrant1, quadrant2]

def process_frames(video_path, num_frames):
    """
    Desc: Captures num_frames frames and returns their average as a grayscale uint8 image.
    Input: video_path (str), num_frames (int).
    Output: numpy.ndarray grayscale or None.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        return None
    return np.mean(frames, axis=0).astype(np.uint8)

def get_central_contour_info(img, tolerance=15):
    """
    Desc: Detects the most central contour in a thresholded image (arena identification).
    Input: img (BGR or grayscale), tolerance (int) distance from center.
    Output: dict {area, centroid, contour} or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_center = np.array([w / 2, h / 2])
    tolerance_zone = [
        (int(img_center[0] + dx), int(img_center[1] + dy))
        for dx in range(-tolerance, tolerance + 1, 5)
        for dy in range(-tolerance, tolerance + 1, 5)
    ]

    most_central_contour = None
    nearest_centroid = None
    min_distance = float('inf')

    for cnt in contours:
        m_moments = cv2.moments(cnt)
        if m_moments["m00"] == 0:
            continue
        cx = int(m_moments["m10"] / m_moments["m00"])
        cy = int(m_moments["m01"] / m_moments["m00"])
        distance = np.linalg.norm(np.array([cx, cy]) - img_center)
        if any(cv2.pointPolygonTest(cnt, pt, False) >= 0 for pt in tolerance_zone):
            if distance < min_distance:
                min_distance = distance
                most_central_contour = cnt
                nearest_centroid = (cx, cy)

    if most_central_contour is not None:
        return {"area": cv2.contourArea(most_central_contour), "centroid": nearest_centroid, "contour": most_central_contour}
    return None

def select_background(video_path):
    """
    Desc: Selects the appropriate reference background image based on the first frame.
    Input: video_path (str).
    Output: str path to selected reference image.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read the first frame of the video.")
        exit()

    img = cv2.imread("C:/TFG_CSIC/Capturas/Captura1.png")
    frame_resized = cv2.resize(frame, (img.shape[1], img.shape[0]))
    video_info = get_central_contour_info(frame_resized)

    if video_info is None:
        return "C:/TFG_CSIC/Capturas/Captura4.png"
    y_coord = video_info["centroid"][1]
    if video_info["area"] > 2500:
        return "C:/TFG_CSIC/Capturas/Captura3.png" if y_coord > 118 else "C:/TFG_CSIC/Capturas/Captura2.png"
    return "C:/TFG_CSIC/Capturas/Captura1.png" if y_coord > 118 else "C:/TFG_CSIC/Capturas/Captura2.png"

def find_floor_square_with_matrix(img, output_size=(400, 400)):
    """
    Desc: Detects the square floor and applies a perspective transform to normalize the arena.
    Input: img (BGR image), output_size (tuple).
    Output: tuple (warped_image, transformation_matrix, bbox).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pts = None
    best_score = -np.inf
    img_h, img_w = gray.shape
    img_center = np.array([img_w / 2, img_h / 2])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        box = np.array(cv2.boxPoints(rect), dtype="float32")
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / min(w, h)
        dist_to_center = np.linalg.norm(np.mean(box, axis=0) - img_center)
        score = -dist_to_center + 0.001 * area
        if ratio < 1.6 and score > best_score:
            best_score = score
            best_pts = box

    if best_pts is None:
        raise ValueError("Square floor not found")

    # Sort points: TL, TR, BR, BL
    pts = sorted(best_pts.tolist(), key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    src_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
    dst_pts = np.array([[0, 0], [output_size[0], 0],
                        [output_size[0], output_size[1]], [0, output_size[1]]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, transform_matrix, output_size)
    bbox = (int(np.min(src_pts[:, 0])), int(np.min(src_pts[:, 1])),
            int(np.max(src_pts[:, 0])), int(np.max(src_pts[:, 1])))
    return warp, transform_matrix, bbox

def find_floor_area_as_mask(img, output_size=(400, 400)):
    """
    Desc: Fallback to identify floor area using contour masks when square detection fails.
    Input: img (BGR image), output_size (tuple).
    Output: tuple (masked_floor_image, area_mask).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    mask = np.zeros_like(gray)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, thickness=1)
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    merged_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not merged_contours:
        raise ValueError("No contours in fallback")

    best_contour = max(merged_contours, key=cv2.contourArea)
    area_mask = np.zeros_like(gray)
    cv2.drawContours(area_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    floor = cv2.bitwise_and(img, img, mask=area_mask)
    return floor, area_mask

def filter_contours_near_crop_border(contours, margin, crop_bbox):
    """
    Desc: Removes contours with ANY point too close to the crop bounding box borders.
    Input: contours (list), margin (int) px buffer, crop_bbox (x_min, y_min, x_max, y_max).
    Output: list of filtered contours.
    """
    x_min, y_min, x_max, y_max = crop_bbox
    filtered = []
    for cnt in contours:
        points = cnt.reshape(-1, 2)
        if not np.any((points[:, 0] < x_min + margin) | (points[:, 0] > x_max - margin) |
                      (points[:, 1] < y_min + margin) | (points[:, 1] > y_max - margin)):
            filtered.append(cnt)
    return filtered

def filter_contours_by_grid(contours, size, origin=(0, 0)):
    """
    Desc: Keeps contours whose centroids fall in corner quadrants {1,3,7,9} of a 3x3 grid.
    Input: contours (list), size (w, h), origin (x, y) offset.
    Output: list of (contour, quadrant_id) tuples.
    """
    w, h = size
    ox, oy = origin
    cell_w, cell_h = w / 3, h / 3
    valid_quadrants = {1, 3, 7, 9}
    result = []
    for cnt in contours:
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        cx_local = int(m['m10'] / m['m00']) - ox
        cy_local = int(m['m01'] / m['m00']) - oy
        if not (0 <= cx_local < w and 0 <= cy_local < h):
            continue
        row = int(cy_local // cell_h) + 1
        col = int(cx_local // cell_w) + 1
        quadrant = (row - 1) * 3 + col
        if quadrant in valid_quadrants:
            result.append((cnt, quadrant))
    return result

def filter_contours_by_radius_ranges(contours_with_quadrants):
    """
    Desc: Selects the best pair of contours based on expected radius ranges.
          Falls back to a "king" heuristic if no typed match is found.
    Input: contours_with_quadrants list of (contour, quadrant_id).
    Output: list of the two best-matching contours.
    """
    ranges = {
        1: lambda r: 9 <= r <= 16,
        2: lambda r: 6 <= r <= 11,
        3: lambda r: 14 <= r <= 19,
        4: lambda r: 8 <= r <= 14,
    }
    data_items = []
    for cnt, quadrant in contours_with_quadrants:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        types = [t_id for t_id, cond in ranges.items() if cond(radius)]
        data_items.append({"contour": cnt, "quadrant": quadrant, "radius": radius, "types": types})

    best_pair, best_sum = [], -1
    for a, b in combinations(data_items, 2):
        if a["quadrant"] == b["quadrant"]:
            continue
        common_types = set(a["types"]).intersection(b["types"])
        for t_type in common_types:
            r_sum = a["radius"] + b["radius"]
            if r_sum > best_sum:
                best_pair = (a, b)
                best_sum = r_sum
    if best_pair:
        return [best_pair[0]["contour"], best_pair[1]["contour"]]

    # Fallback: "king" heuristic (mixed radius pair)
    for a, b in combinations(data_items, 2):
        if a["quadrant"] == b["quadrant"]:
            continue
        r1, r2 = a["radius"], b["radius"]
        if (6 <= r1 <= 8 and 11 <= r2 <= 15) or (6 <= r2 <= 8 and 11 <= r1 <= 15):
            return [a["contour"], b["contour"]]
    return []

def select_first_frames(video_path):
    """
    Desc: Detects objects in first frame; retries with average of first 1000 frames if that fails.
    Input: video_path (str).
    Output: tuple (centers, mean_radius, success_flag).
    """
    background = select_background(video_path)
    img_ref = cv2.imread(background)

    for num_frames in (1, 1000):
        img_test = process_frames(video_path, num_frames)
        cv2.imwrite("img_test.png", img_test)
        img_test = cv2.imread("img_test.png")
        h_orig, w_orig = img_test.shape[:2]
        img_test = cv2.resize(img_test, (img_ref.shape[1], img_ref.shape[0]))
        h_resize, w_resize = img_test.shape[:2]
        scaled_centers, scaled_mean_radius, is_ok = detect_objects_subtract(img_test, img_ref, w_orig, h_orig, w_resize, h_resize)
        if is_ok:
            return scaled_centers, scaled_mean_radius, True
    return scaled_centers, scaled_mean_radius, is_ok

def detect_objects_subtract(img_test, img_ref, w_orig, h_orig, w_resize, h_resize):
    """
    Desc: Subtracts reference background using SSIM to locate the two arena objects.
          Primary path: perspective-warped square floor. Fallback: mask-based alignment.
    Input: img_test (BGR), img_ref (BGR), w_orig, h_orig, w_resize, h_resize (int).
    Output: tuple (scaled_centers, scaled_radius, success_flag).
    """
    try:
        floor_ref, m_matrix_ref, bbox_ref = find_floor_square_with_matrix(img_ref)
        floor_test, m_matrix_test, bbox_test = find_floor_square_with_matrix(img_test)

        gray_ref = cv2.cvtColor(floor_ref, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(floor_test, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.equalizeHist(gray_test)

        score, ssim_map = ssim(gray_ref, gray_test, full=True)
        ssim_diff = ((1 - ssim_map) * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        _, bin_diff = cv2.threshold(ssim_diff, 215, 255, cv2.THRESH_BINARY)
        bin_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, kernel)

        m_inv = np.linalg.inv(m_matrix_test)
        contours, _ = cv2.findContours(bin_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: ALL points must be inside margins
        w_f, h_f = floor_test.shape[1], floor_test.shape[0]
        margin = 10
        filtered_contours = []
        for cnt in contours:
            points = cnt.reshape(-1, 2)
            if np.all((points[:, 0] > margin) & (points[:, 0] < w_f - margin) &
                    (points[:, 1] > margin) & (points[:, 1] < h_f - margin)):
                filtered_contours.append(cnt)

        size_f = (floor_test.shape[1], floor_test.shape[0])
        mask_union = np.zeros_like(bin_diff)
        cv2.drawContours(mask_union, filtered_contours, -1, 255, thickness=cv2.FILLED)
        mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        unified_contours, _ = cv2.findContours(mask_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corner_contours = filter_contours_by_grid(unified_contours, size_f)

        transformed_contours_with_quadrants = []
        for cnt, quadrant in corner_contours:
            if cv2.contourArea(cnt) < 100:
                continue
            cnt_transformed = np.int32(cv2.perspectiveTransform(cnt.astype(np.float32), m_inv))
            transformed_contours_with_quadrants.append((cnt_transformed, quadrant))

        # Single-object recovery: re-crop and re-detect
        if len(transformed_contours_with_quadrants) == 1:
            cnt, quadrant = transformed_contours_with_quadrants[0]
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)

            w_w, h_w = floor_test.shape[1], floor_test.shape[0]
            r_val = min(cx, w_w - cx, cy, h_w - cy) / 2
            x1_o, y1_o = int(r_val), int(r_val)
            x2_o, y2_o = int(w_w - r_val), int(h_w - r_val)
            offset = np.array([[[x1_o, y1_o]]], dtype=np.float32)
            floor_ref_rec = floor_ref[y1_o:y2_o, x1_o:x2_o]
            floor_test_rec = floor_test[y1_o:y2_o, x1_o:x2_o]

            gray_ref_rec = cv2.cvtColor(floor_ref_rec, cv2.COLOR_BGR2GRAY)
            gray_test_rec = cv2.cvtColor(floor_test_rec, cv2.COLOR_BGR2GRAY)
            gray_test_rec = cv2.equalizeHist(gray_test_rec)

            score_rec, ssim_map_rec = ssim(gray_ref_rec, gray_test_rec, full=True)
            ssim_diff_rec = ((1 - ssim_map_rec) * 255).astype(np.uint8)
            _, bin_diff_rec = cv2.threshold(ssim_diff_rec, 247, 255, cv2.THRESH_BINARY)
            bin_diff_rec = cv2.morphologyEx(bin_diff_rec, cv2.MORPH_OPEN, kernel)

            contours_rec, _ = cv2.findContours(bin_diff_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            size_rec = (floor_test_rec.shape[1], floor_test_rec.shape[0])
            mask_union_rec = np.zeros_like(bin_diff_rec)
            cv2.drawContours(mask_union_rec, contours_rec, -1, 255, thickness=cv2.FILLED)
            mask_union_rec = cv2.morphologyEx(mask_union_rec, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

            unified_contours_rec, _ = cv2.findContours(mask_union_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            corner_contours_rec = filter_contours_by_grid(unified_contours_rec, size_rec)

            transformed_contours_with_quadrants = []
            for cnt_r, quad_r in corner_contours_rec:
                if cv2.contourArea(cnt_r) < 100:
                    continue
                cnt_img = cv2.perspectiveTransform(cnt_r.astype(np.float32) + offset, m_inv).astype(np.int32)
                transformed_contours_with_quadrants.append((cnt_img, quad_r))

        # Final selection and scaling
        final_contours = filter_contours_by_radius_ranges(transformed_contours_with_quadrants)
        radii, centers = [], []
        for cnt in final_contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radii.append(radius)
            centers.append((int(x), int(y)))

        mean_radius = np.mean(radii)
        scale_x, scale_y = w_orig / w_resize, h_orig / h_resize
        scaled_centers = [(int(x * scale_x), int(y * scale_y)) for (x, y) in centers]
        scaled_mean_radius = int(mean_radius * (scale_x + scale_y) / 2)
        return scaled_centers, scaled_mean_radius, True

    except (ValueError, TypeError):
        print("Square floor not detected, using inner contour fallback")

    # Fallback path: mask-based alignment
    floor_ref, mask_ref = find_floor_area_as_mask(img_ref)
    floor_test, mask_test = find_floor_area_as_mask(img_test)

    kernel_erode = np.ones((5, 5), np.uint8)
    mask_ref = cv2.erode(mask_ref, kernel_erode, iterations=1)
    mask_test = cv2.erode(mask_test, kernel_erode, iterations=1)

    contours_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_test, _ = cv2.findContours(mask_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def best_contour(contours_list):
        best_c, max_area = None, 0
        for c in contours_list:
            x, y, w, h = cv2.boundingRect(c)
            ratio = max(w, h) / min(w, h)
            area = cv2.contourArea(c)
            if 0.8 < ratio < 1.2 and area > max_area:
                best_c, max_area = c, area
        return best_c if best_c is not None else max(contours_list, key=cv2.contourArea)

    cnt_ref = best_contour(contours_ref)
    cnt_test = best_contour(contours_test)

    def box_from_rect(cnt_obj):
        x, y, w, h = cv2.boundingRect(cnt_obj)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

    h_matrix = cv2.getPerspectiveTransform(box_from_rect(cnt_test), box_from_rect(cnt_ref))
    h_inv = np.linalg.inv(h_matrix)

    floor_test_aligned = cv2.warpPerspective(img_test, h_matrix, (img_ref.shape[1], img_ref.shape[0]))
    floor_ref_cropped = cv2.bitwise_and(img_ref, img_ref, mask=mask_ref)
    floor_test_cropped = cv2.bitwise_and(floor_test_aligned, floor_test_aligned, mask=mask_test)

    gray_ref = cv2.cvtColor(floor_ref_cropped, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(floor_test_cropped, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.equalizeHist(gray_test)

    score, ssim_map = ssim(gray_ref, gray_test, full=True)
    ssim_diff = ((1 - ssim_map) * 255).astype(np.uint8)
    _, bin_diff = cv2.threshold(ssim_diff, 245, 255, cv2.THRESH_BINARY)
    bin_diff = cv2.morphologyEx(bin_diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(bin_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_c, y_c, w_c, h_c = cv2.boundingRect(cnt_test)
    filtered_contours = filter_contours_near_crop_border(contours, 10, (x_c, y_c, x_c + w_c, y_c + h_c))

    mask_union = np.zeros_like(bin_diff)
    cv2.drawContours(mask_union, filtered_contours, -1, 255, thickness=cv2.FILLED)
    mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    unified_contours, _ = cv2.findContours(mask_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_contours = filter_contours_by_grid(unified_contours, (w_c, h_c), origin=(x_c, y_c))

    transformed_contours_with_quadrants = []
    for cnt, quadrant in corner_contours:
        if cv2.contourArea(cnt) < 100:
            continue
        cnt_transformed = np.int32(cv2.perspectiveTransform(cnt.astype(np.float32), h_inv))
        transformed_contours_with_quadrants.append((cnt_transformed, quadrant))

    final_contours = filter_contours_by_radius_ranges(transformed_contours_with_quadrants)
    radii, centers = [], []
    for cnt in final_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radii.append(radius)
        centers.append((int(x), int(y)))

    if len(final_contours) != 2:
        return None, None, False

    mean_radius = np.mean(radii)
    scale_x, scale_y = w_orig / w_resize, h_orig / h_resize
    scaled_centers = [(int(x * scale_x), int(y * scale_y)) for (x, y) in centers]
    scaled_mean_radius = int(mean_radius * (scale_x + scale_y) / 2)
    return scaled_centers, scaled_mean_radius, True

def find_quadrant(object_positions, frame_width, frame_height):
    """
    Desc: Determines in which quadrant of the frame each object is located.
    Input: object_positions list of (x, y), frame_width (int), frame_height (int).
    Output: list of quadrant IDs (1=TL, 2=TR, 3=BL, 4=BR).
    """
    mid_x, mid_y = frame_width // 2, frame_height // 2
    quadrants = []
    for obj_x, obj_y in object_positions:
        if obj_x < mid_x and obj_y < mid_y:
            quadrants.append(1)
        elif obj_x >= mid_x and obj_y < mid_y:
            quadrants.append(2)
        elif obj_x < mid_x and obj_y >= mid_y:
            quadrants.append(3)
        else:
            quadrants.append(4)
    return quadrants

def compute_mouse_centroid(frame_idx, body_part_positions):
    """
    Desc: Computes the centroid of all tracked body keypoints for a given frame.
    Input: frame_idx (int), body_part_positions (dict).
    Output: tuple (centroid_x, centroid_y).
    """
    xs = [pos[0][frame_idx] for pos in body_part_positions.values()]
    ys = [pos[1][frame_idx] for pos in body_part_positions.values()]
    return (np.mean(xs), np.mean(ys))

def is_aligned_via_regression(nose, center_ears, neck, object_center, margin_degrees):
    """
    Desc: Checks if the mouse's head points toward the object using linear regression.
    Input: nose, center_ears, neck, object_center as (x, y) tuples; margin_degrees (float).
    Output: bool True if aligned within margin.
    """
    x_coords_mouse = np.array([neck[0], center_ears[0], nose[0]])
    y_coords_mouse = np.array([neck[1], center_ears[1], nose[1]])

    slope_mouse, _ = np.polyfit(x_coords_mouse, y_coords_mouse, 1)
    mouse_direction = np.array([1, slope_mouse])
    mouse_direction /= np.linalg.norm(mouse_direction)
    if (nose[0] - neck[0]) * mouse_direction[0] + (nose[1] - neck[1]) * mouse_direction[1] < 0:
        mouse_direction = -mouse_direction

    x_coords_object = np.array([center_ears[0], nose[0], object_center[0]])
    y_coords_object = np.array([center_ears[1], nose[1], object_center[1]])

    slope_object, _ = np.polyfit(x_coords_object, y_coords_object, 1)
    object_direction = np.array([1, slope_object])
    object_direction /= np.linalg.norm(object_direction)
    if (object_center[0] - center_ears[0]) * object_direction[0] + (object_center[1] - center_ears[1]) * object_direction[1] < 0:
        object_direction = -object_direction

    cos_angle = np.dot(mouse_direction, object_direction)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle <= np.abs(margin_degrees)

def cleanDataset(csv_path):
    """
    Desc: Reads raw DLC tracking CSV, removes low-likelihood points and interpolates gaps.
    Input: csv_path (str) path to the raw CSV file.
    Output: DataFrame with cleaned and interpolated tracking data.
    """
    df_raw = pd.read_csv(csv_path, header=None, low_memory=False)
    column_names = df_raw.iloc[1] + "_" + df_raw.iloc[2]
    df_raw.columns = column_names
    df = df_raw.drop([0,1,2]).reset_index(drop=True)
    df = df.drop(columns=['bodyparts_coords'])
    df = df.apply(pd.to_numeric, errors='coerce')

    # Compute 10th-percentile likelihood thresholds
    thresholds = {}
    for bp in BODY_PARTS:
        lk = f'{bp}_likelihood'
        if lk in df.columns:
            thresholds[bp] = np.nanpercentile(df[lk], 10)

    # NaN low-likelihood points and interpolate in a single pass
    result = df.copy()
    for bp in BODY_PARTS:
        lk, xc, yc = f'{bp}_likelihood', f'{bp}_x', f'{bp}_y'
        if lk in df.columns:
            result.loc[result[lk] < thresholds.get(bp, 0), [xc, yc]] = np.nan
        result[xc] = result[xc].interpolate(method='linear', limit_direction='both')
        result[yc] = result[yc].interpolate(method='linear', limit_direction='both')
    return result

def explore_directory(directory_path, output_file):
    """
    Desc: Batch-processes all video folders and writes results to CSV.
    Input: directory_path (str), output_file (str).
    Output: None (writes CSV to disk).
    """
    processed_count = 0
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Folder", "TimeObj1", "TimeObj2", "Discrimination_Index_1", "Discrimination_Index_2", "Quadrant1", "Quadrant2"])

        for folder in [f for f in Path(directory_path).iterdir() if f.is_dir()]:
            try:
                tracking_csv = next(folder.glob('*.csv'), None)
                video_mp4 = next(folder.glob('*.mp4'), None)
                if tracking_csv is None or video_mp4 is None:
                    continue

                cap = cv2.VideoCapture(str(video_mp4))
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue

                interpolated_data = cleanDataset(tracking_csv)
                t1, t2, di1, di2, q1, q2 = detect_ROI(interpolated_data, video_mp4)

                # Extract experiment name from folder
                match = re.search(r'Mouse(\d+)-(\d{4}-\d{2}-\d{2})-(\d{2})h(\d{2})m(\d{2})s', folder.name)
                exp_name = f"Mouse{match.group(1)}-{match.group(2)}-{match.group(3)}h{match.group(4)}m{match.group(5)}s" if match else folder.name

                if t1 == -1:
                    writer.writerow([exp_name, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"])
                else:
                    writer.writerow([exp_name, f"{t1:.3f}", f"{t2:.3f}", f"{di1:.3f}", f"{di2:.3f}", q1, q2])

                processed_count += 1
                print(f"Processed folder: {folder}")

            except Exception as e:
                print(f"Error in {folder}: {e}")
                break
    print(f"Processed {processed_count} videos.")

if __name__ == "__main__":
    directory_path = "C:/TFG_CSIC/Videos/NOL_P120"
    last_folder = os.path.basename(directory_path.rstrip("/\\"))
    output_path = f"C:/TFG_CSIC/Outputs_Mej/IntermediateOutput-{last_folder}_Mej_2.csv"
    explore_directory(directory_path, output_path)
    print(f"Results saved to {output_path}")