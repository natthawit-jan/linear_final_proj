import os

import cv2
import pytesseract as psr
import numpy as np
import preprocess as p
import time


def getOnlyCornor(b):
    peri = cv2.arcLength(b, True)
    approx = cv2.approxPolyDP(b, 0.02 * peri, True)
    return approx


## Order the coordinate
def order_coordinate(fourPointArray):
    reshapedPoints = fourPointArray.reshape(4, 2)
    myPointNew = fourPointArray.copy()
    add = reshapedPoints.sum(1)
    myPointNew[0] = reshapedPoints[np.argmin(add)]
    myPointNew[3] = reshapedPoints[np.argmax(add)]
    diff = np.diff(reshapedPoints, axis=1)
    myPointNew[1] = reshapedPoints[np.argmin(diff)]
    myPointNew[2] = reshapedPoints[np.argmax(diff)]
    return myPointNew


## Get each answer row
def split_answer_row(answers_warp_img):
    ten_answers_images = [answers_warp_img[i:i + 100] for i in range(0, 900, 90)]
    return ten_answers_images


def boxes_of_fives(question_img):
    cols = np.hsplit(question_img, 5)
    return cols


def get_lst_of_answer(questions):
    '''
    :param questions: a list of question row
    :return: list of the answer [A, B, A]
    '''
    ans = []
    for row in questions:
        eachBox = np.array([np.count_nonzero(box) for box in boxes_of_fives(row)])
        ans_key = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        ans.append(ans_key[np.argmax(eachBox)])
    return np.array(ans)


FILENAME = 'exam.jpg'

cap = cv2.VideoCapture(0)
KEY_ANSWER = np.array(['A', 'B', 'B', 'A', 'E', 'D', 'B', 'B', 'C', 'E'])
W, H = 1080, 1920
while (True):
    ret, frame = cap.read()

    try:
        imgContours = frame.copy()
        onlyBoxesImg = frame.copy()
        imgFinal = frame.copy()
        imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2BGRA)

        ## Preprocessing
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        edge = cv2.Canny(imgBlur, 50, 20)

        ## Find the contour of the reactangle
        contours, hir = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 10)
        # cv2.imshow('contours', imgContours)
        ## For each contour we find the 3, boxes
        contourLst = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 40:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    contourLst.append(contour)
        contourLst = sorted(contourLst, key=cv2.contourArea, reverse=True)

        boxes = list(map(getOnlyCornor, contourLst))

        questionBox, idBox, scoreBox = boxes[0], boxes[1], boxes[2]

        if questionBox.size != 0 and scoreBox.size != 0 and idBox.size != 0:
            cv2.drawContours(onlyBoxesImg, questionBox, -1, (0, 0, 255), 20)
            cv2.drawContours(onlyBoxesImg, scoreBox, -1, (0, 0, 255), 20)
            cv2.drawContours(onlyBoxesImg, idBox, -1, (0, 0, 255), 20)

            questionBox = order_coordinate(questionBox)
            idBox = order_coordinate(idBox)
            scoreBox = order_coordinate(scoreBox)

            ## Show warp
            pt1 = np.float32(questionBox)
            pt2 = np.float32([[0, 0], [500, 0], [0, 1000], [500, 1000]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpedColored = cv2.warpPerspective(frame, matrix, (500, 1000))
            imgWarpedGrey = cv2.cvtColor(imgWarpedColored, cv2.COLOR_BGR2GRAY)
            imgTresh = cv2.threshold(imgWarpedGrey, 170, 255, cv2.THRESH_BINARY_INV)[1]

            pt1Score = np.float32(scoreBox)
            pt2Score = np.float32([[0, 0], [500, 0], [0, 320], [500, 320]])
            matrixGrade = cv2.getPerspectiveTransform(pt1Score, pt2Score)
            ScoreWarpedColoredImg = cv2.warpPerspective(frame, matrixGrade, (500, 320))

            ## GET THE ANSWERS
            questions = split_answer_row(imgTresh)

            answers_detected = get_lst_of_answer(questions)

            SCORE = (np.count_nonzero(KEY_ANSWER == answers_detected) / 10.) * 100.
            DISPLAY_SCORE = f'SCORE = {SCORE} %'

            # setup text
            font = cv2.QT_FONT_NORMAL

            cv2.putText(imgFinal, DISPLAY_SCORE, (100, 170), font, 3, (255, 255, 0), 4)

            cv2.imshow('frame', imgFinal)
    except:
        cv2.imshow('frame', frame)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        print('signal sent : stopping')
        break
