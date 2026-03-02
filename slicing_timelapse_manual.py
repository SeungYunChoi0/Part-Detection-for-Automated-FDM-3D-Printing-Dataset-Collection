import cv2
import os

# ==========================================
# 1. 사용자 설정
# ==========================================
VIDEO_PATH = '/Volumes/Yun_ssd/Modulino_CV/dataset_video_white.mp4'  # 영상 파일 경로
SAVE_DIR = 'dataset_norm_white'  # 저장 폴더
FILE_PREFIX = 'bambu_p1s'      # 파일명 앞부분 (xxx)

# ==========================================
# 2. 메인 클래스 및 함수
# ==========================================

def point_labeling_tool():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 30fps 기준 1프레임은 약 0.033초이므로, 
    # 0.01초 전후는 인접한 이전/다음 프레임을 잡는 로직으로 구현됩니다.
    marks = [] # [[t_prev, t_curr, t_next], layer_label] 저장 리스트
    paused = False
    
    win_name = "3D Printing Point Collector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"\n--- [조작 방법] ---")
    print(f"Space : 일시정지 / 재생")
    print(f"A / D : 1초 뒤로 / 앞으로 이동")
    print(f"E : 현재 시점 마킹 (해당 시점 + 전후 0.01초 추출)")
    print(f"ESC : 마킹 종료 및 이미지 일괄 생성")
    print(f"-------------------\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
        
        curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        curr_time = curr_frame / fps

        # 화면 오버레이 표시
        display_frame = frame.copy()
        status = "PAUSED" if paused else "PLAYING"
        
        cv2.putText(display_frame, f"Time: {curr_time:.3f}s | {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Total Marks: {len(marks)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 27: # ESC
            break
        elif key == ord(' '): # Space
            paused = not paused
        elif key == ord('a'): # A
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_frame - int(fps)))
            paused = True # 이동 후 정지
        elif key == ord('d'): # D
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames-1, curr_frame + int(fps)))
            paused = True # 이동 후 정지
        
        elif key == ord('e'): # E: 현재 시점 마킹
            # 0.01초 전, 현재, 0.01초 후 시간 계산
            t_curr = curr_time
            t_prev = max(0, t_curr - 0.01)
            t_next = min(total_frames/fps, t_curr + 0.01)
            
            print(f"📍 시점 포착: {t_curr:.3f}s")
            
            # 레이어 입력
            layer_num = input(f"📝 레이어 번호 입력 (L?): ")
            layer_label = f"L{layer_num}"
            
            marks.append([[t_prev, t_curr, t_next], layer_label])
            print(f"✅ 리스트 추가 완료: {layer_label} (3프레임 추출 대기)")

    cap.release()
    cv2.destroyAllWindows()

    # ==========================================
    # 3. 이미지 일괄 생성 (촤르륵)
    # ==========================================
    if not marks:
        print("추출할 데이터가 없습니다.")
        return

    print(f"\n🚀 총 {len(marks) * 3}장의 이미지 추출을 시작합니다...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    sequence_num = 1

    for time_triple, layer in marks:
        for t in time_triple:
            target_f = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, frame = cap.read()
            if not ret: continue

            # 파일명 규칙: xxx_001_L1.jpg
            file_name = f"{FILE_PREFIX}_{str(sequence_num).zfill(3)}_{layer}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, file_name), frame)
            
            print(f"📸 저장 완료: {file_name} (at {t:.3f}s)")
            sequence_num += 1

    cap.release()
    print(f"\n✨ 작업 완료! 저장 위치: {SAVE_DIR}")

if __name__ == "__main__":
    point_labeling_tool()