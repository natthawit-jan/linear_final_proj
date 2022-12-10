import cv2
import numpy as np
import utils

cap = cv2.VideoCapture(0)
KEY_ANSWER = np.array(['A', 'D', 'D', 'B', 'C', 'C', 'E', 'E', 'D', 'D'])
ANS_SET = 'ABCDE'
W, H = 1080, 1920
while True:
    ret, frame = cap.read()

    try:
        imgContours = frame.copy()
        onlyBoxesImg = frame.copy()
        imgFinal = frame.copy()
        imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2BGRA)

        # Preprocessing
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the image to greyscale
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Blur image a little
        edge = cv2.Canny(imgBlur, 50, 20)  # Get the edge we want

        # Find the contour around all edges detected
        contours, hir = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 10)

        contourLst = []
        for contour in contours:
            # For all the contours, we are only interested in 2 boxes (Questions and Score boxes)
            # These two boxes are supposed to have big areas than the others, we guessed that they're around 5,000
            area = cv2.contourArea(contour)
            if area > 5000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, (peri * (3/100.)), True)
                # Check if we have four corners, if yes, then this is our boxes
                if len(approx) == 4:
                    contourLst.append(contour)

        contourLst = sorted(contourLst, key=cv2.contourArea, reverse=True)
        boxes = list(map(utils.getOnlyCornor, contourLst))
        questionBox, scoreBox = boxes[0], boxes[1]

        if questionBox.size != 0 and scoreBox.size != 0:
            cv2.drawContours(onlyBoxesImg, questionBox, -1, (0, 0, 255), 20)
            cv2.drawContours(onlyBoxesImg, scoreBox, -1, (0, 0, 255), 20)

            questionBox = utils.order_coordinate(questionBox)
            scoreBox = utils.order_coordinate(scoreBox)

            # Show warp
            pt1 = np.float32(questionBox)
            pt2 = np.float32([[0, 0], [500, 0], [0, 1000], [500, 1000]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpedColored = cv2.warpPerspective(frame, matrix, (500, 1000))

            imgWarpedGrey = cv2.cvtColor(imgWarpedColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpedGrey, 170, 255, cv2.THRESH_BINARY_INV)[1]

            pt1Score = np.float32(scoreBox)
            pt2Score = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
            matrixGrade = cv2.getPerspectiveTransform(pt1Score, pt2Score)
            ScoreWarpedColoredImg = cv2.warpPerspective(frame, matrixGrade, (500, 500))

            # Split the full image into 10 equally-sized images
            questions = utils.split_answer_row(imgThresh)

            # Get list of marks
            answers_detected = utils.get_lst_of_answer(questions)

            SCORE = (np.count_nonzero(KEY_ANSWER == answers_detected) / 10.) * 100.
            DISPLAY_SCORE = f'{SCORE} %'


            font = cv2.QT_FONT_NORMAL

            # Question Sheet
            imgRawSheet = np.zeros_like(imgWarpedColored)
            utils.draw_circles(imgRawSheet, answers_detected, KEY_ANSWER, ANS_SET)
            invMatrixSheet = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvSheetDisplay = cv2.warpPerspective(imgRawSheet, invMatrixSheet, (H, W))
            imgInvSheetDisplay = utils.convert_to_transparent(imgInvSheetDisplay)

            imgRawScore = np.zeros_like(ScoreWarpedColoredImg)
            cv2.putText(imgRawScore, DISPLAY_SCORE, (100, 200), font, 3, (0, 145, 255), 5)
            invMatrixScore = cv2.getPerspectiveTransform(pt2Score, pt1Score)
            imgInvScoreDisplay = cv2.warpPerspective(imgRawScore, invMatrixScore, (H, W))

            imgInvScoreDisplay = utils.convert_to_transparent(imgInvScoreDisplay)

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvSheetDisplay, .9, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvScoreDisplay, .9, 1, 0)

            cv2.imshow('frame', imgFinal)
        else:
            cv2.imshow('frame', imgFinal)

    except Exception as e:
        cv2.imshow('frame', frame)
        # print(e)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        print('signal sent : stopping')
        break
