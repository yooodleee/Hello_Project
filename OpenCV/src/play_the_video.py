# 동영상 재생하기

import cv2 as cv

# VideoCapture의 argument로 동영상 파일을 넣는다.
cap = cv.VideoCapture('output.avi')

if cap.isOpened() == False:
    print("동영상을 열 수 없습니다.")
    exit(1)

while(True):

    ret, img_frame = cap.read()

    # 동영상 끝까지 재생하면 read 함수는 False를 리턴함.
    if ret == False:
        print("동영상 파일 읽기 완료")
        break

    cv.imshow('Color', img_frame)

    # 동영상 재생 속도를 조정하기 위해 waitKey 함수를 설정.
    key = cv.waitKey(25)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()