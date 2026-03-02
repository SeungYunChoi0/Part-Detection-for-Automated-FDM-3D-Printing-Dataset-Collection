#!/usr/bin/env python3
"""
Batch Cropping Script for White Parts
흰색 출력물 일괄 크로핑 스크립트

사용법:
    python batch_crop_white.py

입력: /Volumes/Yun_ssd/Modulino_CV/data_norm_white
출력: /Volumes/Yun_ssd/Modulino_CV/data_norm_white_cropped
"""

import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json
import sys

# ============================================================
# 설정
# ============================================================
INPUT_DIR = Path('/Volumes/Yun_ssd/Modulino_CV/blue/dataset_norm_blue')
OUTPUT_DIR = Path('/Volumes/Yun_ssd/Modulino_CV/data_norm_white_cropped_FF_blue')
FAILED_DIR = OUTPUT_DIR / 'failed'
LOG_FILE = OUTPUT_DIR / 'processing_log.txt'
SUMMARY_FILE = OUTPUT_DIR / 'summary.json'


LOWER_WHITE = np.array([0, 0, 150]) 
UPPER_WHITE = np.array([179, 35, 255])

# 필터링 파라미터
MIN_AREA = 3000  # 5000 → 3000 (원복)
MAX_AREA = 200000  # 100000 → 200000 (완화)
MIN_ASPECT_RATIO = 1.0
MAX_ASPECT_RATIO = 6.0
MIN_EXTENT = 0.25
CENTER_REGION = 0.8

# 영역 제외 (완화)
HEAD_REGION_X_MAX = 0.25
HEAD_REGION_Y_MAX = 0.30
BOTTOM_REGION_Y_MIN = 0.75  # 0.65 → 0.75 (완화)

# ============================================================
# 출력물 검출 함수
# ============================================================
def detect_white_part(img):
    """
    흰색 출력물 검출 (HSV + 구멍 패턴)
    
    Returns:
        bbox (x, y, w, h) or None if failed
    """
    height, width = img.shape[:2]
    
    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 색상 마스킹
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    
    # 모폴로지 연산 (강화 - 전체 영역 캐치)
    kernel = np.ones((7,7), np.uint8)
    
    # Closing: 내부 구멍 메우기 (강화!)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Opening: 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilation: 경계 확장 (전체 영역 캐치)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 컨투어 찾기
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 유효한 객체 필터링
    valid_objects = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / float(min(w, h))
            
            # 헤드 영역 제외 (왼쪽 상단)
            in_head_region = (x < width * HEAD_REGION_X_MAX and y < height * HEAD_REGION_Y_MAX)
            if in_head_region:
                continue
            
            # 하단 영역 제외 (베드 가장자리)
            center_y = y + h/2
            in_bottom_region = (center_y > height * BOTTOM_REGION_Y_MIN)
            if in_bottom_region:
                continue
            
            # 중앙 영역 체크
            center_x = x + w/2
            center_y = y + h/2
            margin = (1 - CENTER_REGION) / 2
            in_center = (width*margin < center_x < width*(1-margin) and 
                        height*margin < center_y < height*(1-margin))
            
            rect_area = w * h
            extent = area / float(rect_area)
            
            if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and extent > MIN_EXTENT:
                # 구멍 개수 세기
                num_holes = 0
                if hierarchy is not None and i < len(hierarchy[0]):
                    child_idx = hierarchy[0][i][2]
                    while child_idx != -1:
                        num_holes += 1
                        if child_idx < len(hierarchy[0]):
                            child_idx = hierarchy[0][child_idx][0]
                        else:
                            break
                
                valid_objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'holes': num_holes,
                    'in_center': in_center
                })
    
    if len(valid_objects) == 0:
        return None
    
    # 중앙 영역 우선
    center_objects = [obj for obj in valid_objects if obj['in_center']]
    if len(center_objects) == 0:
        center_objects = valid_objects
    
    # 구멍 + 면적 우선순위 정렬
    center_objects.sort(key=lambda x: (x['holes'] > 0, x['area']), reverse=True)
    
    return center_objects[0]['bbox']


def crop_and_enhance(img, bbox):
    """
    이미지 크로핑 및 후처리
    
    Args:
        img: 원본 이미지
        bbox: (x, y, w, h)
    
    Returns:
        cropped and enhanced image
    """
    height, width = img.shape[:2]
    x, y, w, h = bbox
    
    # 여백 추가
    margin = int(min(w, h) * 0.15)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)
    
    # 크로핑
    cropped = img[y1:y2, x1:x2].copy()
    
    # 빛 반사 제거
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    
    # 감마 보정
    gamma = 0.9
    l_gamma = np.array(255 * (l_eq / 255) ** gamma, dtype='uint8')
    
    # 재결합
    lab_final = cv2.merge([l_gamma, a, b])
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    # 디노이징
    result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)
    
    # 샤프닝
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel_sharp)
    
    return result


# ============================================================
# 메인 처리 함수
# ============================================================
def process_images():
    """
    모든 이미지 일괄 처리
    """
    print(f"{'='*60}")
    print(f"흰색 출력물 일괄 크로핑")
    print(f"{'='*60}")
    print(f"입력 폴더: {INPUT_DIR}")
    
    # 입력 폴더 존재 확인
    if not INPUT_DIR.exists():
        print(f"\n❌ 오류: 입력 폴더를 찾을 수 없습니다!")
        print(f"경로: {INPUT_DIR}")
        print(f"\n폴더 경로를 확인해주세요.")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 목록
    print(f"\n이미지 파일 검색 중...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        found = list(INPUT_DIR.glob(f'*{ext}'))
        if found:
            print(f"  {ext}: {len(found)}개")
        image_files.extend(found)
    
    image_files = sorted(image_files)
    total_images = len(image_files)
    
    # 이미지 개수 확인
    if total_images == 0:
        print(f"\n❌ 오류: 이미지 파일을 찾을 수 없습니다!")
        print(f"\n입력 폴더: {INPUT_DIR}")
        print(f"지원 확장자: {', '.join(image_extensions)}")
        sys.exit(1)
    
    print(f"\n출력 폴더: {OUTPUT_DIR}")
    print(f"총 이미지 수: {total_images}개")
    print(f"{'='*60}\n")
    
    # 통계 초기화
    stats = {
        'total': total_images,
        'success': 0,
        'failed': 0,
        'failed_files': [],
        'start_time': datetime.now().isoformat(),
        'processing_times': []
    }
    
    # 로그 파일 초기화
    with open(LOG_FILE, 'w') as f:
        f.write(f"Processing started at {stats['start_time']}\n")
        f.write(f"Total images: {total_images}\n")
        f.write("="*60 + "\n\n")
    
    # 처리 시작
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        img_start = time.time()
        
        try:
            # 이미지 읽기
            img = cv2.imread(str(img_path))
            
            if img is None:
                raise ValueError(f"Failed to read image: {img_path.name}")
            
            # 출력물 검출
            bbox = detect_white_part(img)
            
            if bbox is None:
                # 검출 실패
                stats['failed'] += 1
                stats['failed_files'].append(img_path.name)
                
                # 실패 이미지 복사 (디버깅용)
                failed_path = FAILED_DIR / img_path.name
                cv2.imwrite(str(failed_path), img)
                
                # 로그 기록
                with open(LOG_FILE, 'a') as f:
                    f.write(f"[FAILED] {img_path.name} - No part detected\n")
                
                print(f"[{idx}/{total_images}] ❌ FAILED: {img_path.name}")
                
            else:
                # 크로핑 및 후처리
                cropped = crop_and_enhance(img, bbox)
                
                # 저장
                output_path = OUTPUT_DIR / img_path.name
                cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
                
                stats['success'] += 1
                
                # 처리 시간 기록
                img_time = time.time() - img_start
                stats['processing_times'].append(img_time)
                
                # 진행률 표시
                progress = (idx / total_images) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / idx) * (total_images - idx)
                
                print(f"[{idx}/{total_images}] ✅ SUCCESS: {img_path.name} "
                      f"({img_time:.2f}s) - {progress:.1f}% "
                      f"(ETA: {eta/60:.1f}min)")
        
        except Exception as e:
            # 예외 발생
            stats['failed'] += 1
            stats['failed_files'].append(img_path.name)
            
            with open(LOG_FILE, 'a') as f:
                f.write(f"[ERROR] {img_path.name} - {str(e)}\n")
            
            print(f"[{idx}/{total_images}] ⚠️ ERROR: {img_path.name} - {str(e)}")
    
    # 총 처리 시간
    total_time = time.time() - start_time
    stats['end_time'] = datetime.now().isoformat()
    stats['total_time_seconds'] = total_time
    stats['avg_time_per_image'] = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    
    # 최종 요약
    print(f"\n{'='*60}")
    print(f"처리 완료!")
    print(f"{'='*60}")
    print(f"총 처리 시간: {total_time/60:.2f}분 ({total_time:.1f}초)")
    
    if total_images > 0:
        success_rate = (stats['success']/total_images*100)
        print(f"성공: {stats['success']}개 ({success_rate:.1f}%)")
        print(f"실패: {stats['failed']}개 ({(stats['failed']/total_images*100):.1f}%)")
    
    print(f"평균 처리 시간: {stats['avg_time_per_image']:.3f}초/이미지")
    print(f"\n출력 폴더: {OUTPUT_DIR}")
    print(f"실패 이미지: {FAILED_DIR}")
    print(f"로그 파일: {LOG_FILE}")
    print(f"{'='*60}")
    
    # 실패한 파일 목록 출력
    if stats['failed'] > 0:
        print(f"\n실패한 파일 ({stats['failed']}개):")
        for failed_file in stats['failed_files'][:10]:  # 최대 10개만 표시
            print(f"  - {failed_file}")
        if stats['failed'] > 10:
            print(f"  ... 외 {stats['failed']-10}개")
    
    # JSON 요약 저장
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n요약 파일: {SUMMARY_FILE}")
    
    return stats


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    try:
        stats = process_images()
        
        # 성공률이 90% 미만이면 경고
        if stats['total'] > 0 and (stats['success'] / stats['total']) < 0.9:
            print("\n⚠️ 경고: 성공률이 90% 미만입니다. 파라미터 조정이 필요할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)