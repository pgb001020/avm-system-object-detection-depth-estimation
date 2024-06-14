import torch
from depth_anything.dpt import DepthAnything
import os
import cv2

def main(encoder='vitl', img_path='assets/examples', outdir='depth_vis'):
    # 모델 초기화
    depth_predictor = DepthAnything(encoder_type=encoder)

    # 이미지 파일 목록 가져오기
    img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    
    # 출력 디렉토리 생성
    os.makedirs(outdir, exist_ok=True)

    for img_file in img_files:
        # 이미지 로드
        img = cv2.imread(os.path.join(img_path, img_file))
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue

        # 깊이 예측
        depth_map = depth_predictor.predict(img)

        # 결과 저장
        output_file = os.path.join(outdir, os.path.splitext(img_file)[0] + '_depth.png')
        cv2.imwrite(output_file, depth_map)
        print(f"Saved depth map to {output_file}")

if __name__ == "__main__":
    main()
