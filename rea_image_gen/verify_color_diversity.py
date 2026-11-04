import os
import sys
import argparse
from PIL import Image
import numpy as np

def analyze_image_color_diversity(image_path: str, verbose: bool = True) -> dict:
    """
    생성된 이미지를 분석하여 단색/노이즈/깨진 이미지 여부를 판단하고 색상 다양성을 평가합니다.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    total_pixels = width * height
    arr = np.array(img)

    # 1. 고유 색상(Unique colors) 수 및 비율
    pixels_flat = arr.reshape(-1, 3)
    unique_colors = len(np.unique(pixels_flat, axis=0))
    unique_color_ratio = unique_colors / total_pixels

    # 2. 채널별 평균 및 표준편차 (색상 분포 및 대비)
    mean_per_channel = arr.mean(axis=(0, 1))
    std_per_channel = arr.std(axis=(0, 1))
    avg_std = float(np.mean(std_per_channel))

    # 3. 색상 엔트로피 (Color Entropy via 3D RGB histogram 16x16x16 bins = 4096 bins)
    bins = 16
    hist, _ = np.histogramdd(pixels_flat, bins=(bins, bins, bins), range=((0, 256), (0, 256), (0, 256)))
    hist_prob = hist.flatten() / total_pixels
    hist_prob = hist_prob[hist_prob > 0]
    entropy = float(-np.sum(hist_prob * np.log2(hist_prob)))
    max_possible_entropy = float(np.log2(bins ** 3))  # 12.0 for 16^3 bins

    # 4. 종합 판정 기준
    # - 단색/블랙스크린/화이트스크린: std < 10 or unique_colors < 100
    # - 흐릿하거나 단순 단조로운 그래디언트: entropy < 3.0
    # - 정상적이고 색상이 풍부한 생성 이미지: entropy >= 4.0 and avg_std >= 25.0 and unique_colors >= 1000
    is_valid = True
    issues = []

    if avg_std < 15.0:
        is_valid = False
        issues.append(f"색상 표준편차가 너무 낮음 (avg std: {avg_std:.2f} < 15.0) - 단색 화면 의심")
    if unique_colors < 500:
        is_valid = False
        issues.append(f"고유 색상 수가 너무 적음 ({unique_colors} < 500) - 렌더링 실패 의심")
    if entropy < 3.5:
        is_valid = False
        issues.append(f"색상 엔트로피가 너무 낮음 ({entropy:.2f} < 3.5) - 색상 다양성 부족")

    quality_score = min(100.0, (entropy / max_possible_entropy * 50.0) + (min(avg_std, 70.0) / 70.0 * 50.0))

    result = {
        "image_path": image_path,
        "resolution": f"{width}x{height}",
        "total_pixels": total_pixels,
        "unique_colors": unique_colors,
        "unique_color_ratio": round(unique_color_ratio, 4),
        "channel_std": {
            "R": round(float(std_per_channel[0]), 2),
            "G": round(float(std_per_channel[1]), 2),
            "B": round(float(std_per_channel[2]), 2),
            "avg": round(avg_std, 2)
        },
        "color_entropy": round(entropy, 2),
        "max_entropy": round(max_possible_entropy, 2),
        "quality_score": round(quality_score, 1),
        "is_valid": is_valid,
        "issues": issues
    }

    if verbose:
        print("=" * 60)
        print(f"[이미지 색상 다양성 분석 결과] {image_path}")
        print("=" * 60)
        print(f"- 해상도: {result['resolution']} ({result['total_pixels']:,} 픽셀)")
        print(f"- 고유 색상 개수: {result['unique_colors']:,}개 (전체 픽셀 대비 {result['unique_color_ratio']*100:.2f}%)")
        print(f"- 채널별 표준편차: R={result['channel_std']['R']}, G={result['channel_std']['G']}, B={result['channel_std']['B']} (평균: {result['channel_std']['avg']})")
        print(f"- 색상 엔트로피: {result['color_entropy']} / {result['max_entropy']}")
        print(f"- 색상 다양성 점수: {result['quality_score']:.1f} / 100.0")
        print("-" * 60)
        if result['is_valid']:
            print("✅ 판정 결과: [PASS] 정상적으로 다양한 색상이 분포된 완성된 이미지입니다.")
        else:
            print(f"❌ 판정 결과: [FAIL] 비정상 이미지 감지 - {', '.join(issues)}")
        print("=" * 60)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="생성된 이미지 색상 다양성 검증")
    parser.add_argument("image_path", help="검증할 이미지 파일 경로")
    args = parser.parse_args()

    analyze_image_color_diversity(args.image_path)

