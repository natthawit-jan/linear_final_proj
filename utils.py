import cv2
import pytesseract as psr
import numpy as np


## Draw a circle for each question
def draw_circles(img, real_answers, KEY_ANSWER, ANS_SET):
    question_cnt = 0

    def _draw_green(img, pt):
        cv2.circle(img, pt, 20, (0, 255, 0),  cv2.FILLED)

    def _draw_red(img, pt):
        cv2.circle(img, pt, 20, (0, 0, 255), cv2.FILLED)

    for question in range(0, 1000, 100):
        ans = KEY_ANSWER[question_cnt]
        ind = 0
        middle_h = question + 50

        pnt_lst = []
        for choice in range(0, 500, 100):
            middle_w = choice + 50

            pnt_lst.append((middle_w, middle_h))
        ind = ANS_SET.find(ans)
        if real_answers[question_cnt] == '-':
            pass
        elif ans == real_answers[question_cnt]:
            _draw_green(img, pnt_lst[ind])
        else:
            # Draw on the answer
            _draw_green(img, pnt_lst[ind])
            # Draw on the correct answer
            ind = ANS_SET.find(real_answers[question_cnt])
            _draw_red(img, pnt_lst[ind])
        question_cnt += 1


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
    ten_answers_images = [answers_warp_img[i:i + 100] for i in range(0, 1000, 100)]
    # for num, i in enumerate(ten_answers_images):
    #     cv2.imshow(str(num) + '_row ', i)
    return ten_answers_images


def boxes_of_fives(question_img):
    cols = np.hsplit(question_img, 5)
    # ind = 1
    # for i in cols:
    #     cv2.imshow('first '  + str(ind), i)
    #     ind += 1
    return cols





def get_lst_of_answer(questions):
    '''
    :param questions: a list of question row
    :return: list of the answer [A, B, A]
    '''
    ans = []
    ind = 1
    for row in questions:
        cv2.imshow(f'row{ind}', row)
        eachBox = np.array([np.count_nonzero(box)/box.size for box in boxes_of_fives(row)])
        # The marked one will content at least 30% of the white pixels
        print(f'Row {ind}', eachBox)
        marks = np.count_nonzero(eachBox >= .25)
        if marks >= 2 or marks == 0:
            ans.append('-')
        else:
            ans_key = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            ans.append(ans_key[np.argmax(eachBox)])
        ind += 1
    return np.array(ans)


def convert_to_transparent(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)

    rgba = [b, g, r, alpha]
    img = cv2.merge(rgba, 4)
    return img
