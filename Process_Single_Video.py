"""
Single-video processor for NOL task -- debug/interactive version.
Imports all shared logic from Codigo_final and runs with display_video=True.
"""

from Codigo_final_Mej import cleanDataset, detect_ROI


if __name__ == "__main__":
    csv_file = (
        "C:/TFG_CSIC/Videos/NOL_P75/Mouse4163-2024-07-11-11h03m47s-dshow/Mouse4163-2024-07-11-11h03m47s-dshow___--convertDLC_resnet50_DLC_NOL_2021_microscopeDec29shuffle3_100000.csv"
    )
    video_file = (
      "C:/TFG_CSIC/Videos/NOL_P75/Mouse4163-2024-07-11-11h03m47s-dshow/Mouse4163-2024-07-11-11h03m47s-dshow___--convert.mp4"
    )

    interpolated = cleanDataset(csv_file)
    time1, time2, di1, di2, q1, q2 = detect_ROI(
        interpolated, video_file, display_video=True)

    print(f"Object 1 -- time: {time1:.3f}s, DI: {di1:.3f}, quadrant: {q1}")
    print(f"Object 2 -- time: {time2:.3f}s, DI: {di2:.3f}, quadrant: {q2}")
