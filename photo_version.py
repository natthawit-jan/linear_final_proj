import cv2
import numpy as np
import utils

# cap = cv2.VideoCapture(0)

# Define the master answer
KEY_ANSWER = np.array(['A', 'D', 'D', 'B', 'C', 'C', 'E', 'E', 'D', 'D'])


ANS_SET = 'ABCDE'

W, H = 2000, 1920
frame = cv2.imread('EXAM_with_answer.jpg')
frame = cv2.resize(frame, (H, W))
imgContours = frame.copy()
onlyBoxesImg = frame.copy()
imgFinal = frame.copy()
imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2BGRA)

# Preprocessing
imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the image to greyscale
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Blur image a little
edge = cv2.Canny(imgBlur, 50, 20) # Get the edge we want

cv2.imshow('edge', edge)




## Find the contour of the reactangle
contours, hir = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 10)
contourLst = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 5000:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            # print(area)
            contourLst.append(contour)

contourLst = sorted(contourLst, key=cv2.contourArea, reverse=True)
boxes = list(map(utils.getOnlyCornor, contourLst))
questionBox, scoreBox = boxes[0], boxes[1]

if questionBox.size != 0 and scoreBox.size != 0:
    cv2.drawContours(onlyBoxesImg, questionBox, -1, (0, 0, 255), 20)
    cv2.drawContours(onlyBoxesImg, scoreBox, -1, (0, 0, 255), 20)


    questionBox = utils.order_coordinate(questionBox)
    scoreBox = utils.order_coordinate(scoreBox)

    ## Show warp
    pt1 = np.float32(questionBox)
    pt2 = np.float32([[0, 0], [500, 0], [0, 1000], [500, 1000]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpedColored = cv2.warpPerspective(frame, matrix, (500, 1000))
    imgWarpedGrey = cv2.cvtColor(imgWarpedColored, cv2.COLOR_BGR2GRAY)
    imgTresh = cv2.threshold(imgWarpedGrey, 170, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('imgtr', imgTresh)

    # cv2.imshow('warp', imgTresh)
    pt1Score = np.float32(scoreBox)
    pt2Score = np.float32([[0, 0], [500, 0], [0, 320], [500, 320]])
    matrixGrade = cv2.getPerspectiveTransform(pt1Score, pt2Score)
    ScoreWarpedColoredImg = cv2.warpPerspective(frame, matrixGrade, (500, 320))

    cv2.imshow('SCOREEEE', ScoreWarpedColoredImg)

    ## GET THE ANSWERS
    questions = utils.split_answer_row(imgTresh)
    print(questions)

    answers_detected = utils.get_lst_of_answer(questions)
    SCORE = (np.count_nonzero(KEY_ANSWER == answers_detected) / 10.) * 100.
    DISPLAY_SCORE = f'{SCORE} %'

    font = cv2.QT_FONT_NORMAL

    ## QUESTION SHEET
    imgRawSheet = np.zeros_like(imgWarpedColored)
    utils.draw_circles(imgRawSheet, answers_detected, KEY_ANSWER, ANS_SET)
    invMatrixSheet = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvSheetDisplay = cv2.warpPerspective(imgRawSheet, invMatrixSheet, (H, W))

    imgInvSheetDisplay = utils.convert_to_transparent(imgInvSheetDisplay)

    imgRawScore = np.zeros_like(ScoreWarpedColoredImg)
    cv2.putText(imgRawScore, DISPLAY_SCORE, (100, 170), font, 3, (0, 0, 255), 4)
    invMatrixScore = cv2.getPerspectiveTransform(pt2Score, pt1Score)
    imgInvScoreDisplay = cv2.warpPerspective(imgRawScore, invMatrixScore, (H, W))

    cv2.imshow('SCPRE WITH TEXT', imgInvScoreDisplay)

    ## Convert to transparant
    imgInvScoreDisplay = utils.convert_to_transparent(imgInvScoreDisplay)

    cv2.putText(imgFinal, DISPLAY_SCORE, (100, 170), font, 3, (255, 255, 0), 4)

    imgFinal = cv2.addWeighted(imgFinal, .7, imgInvScoreDisplay, .6, 1, 0)
    # imgFinal = cv2.add(imgFinal, imgInvScoreDisplay)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvSheetDisplay, .9, 1, 0)

    cv2.imshow('frame', imgFinal)

    cv2.waitKey(0)
