import cv2
import pytesseract
import os
import matplotlib.pyplot as plt


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.basename("gray.png"), gray)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # # Plot histogram
    # plt.plot(hist, color='black')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.title('Grayscale Image Histogram')
    # plt.show()

    blur_image = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2)
    cv2.imwrite(os.path.basename("thresh.png"), thresh)
    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contoured_img = cv2.drawContours(thresh, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.basename("contours.png"), contoured_img)

    return contours

def filter_contours(contours, min_area=100):
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            filtered_contours.append(contour)
    return filtered_contours

def extract_digits(image, contours):
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_roi = image[y:y+h, x:x+w]
        digits.append(digit_roi)
    return digits

def recognize_digits(digits):
    recognized_digits = []
    for digit in digits:
        text = pytesseract.image_to_string(digit, config='--psm 6')
        recognized_digits.append(text.strip())
    return recognized_digits

image = cv2.imread("3.jpg")

thresh = preprocess_image(image)

contours = find_contours(thresh)

filtered_contours = filter_contours(contours)

digits = extract_digits(image, filtered_contours)

recognized_digits = recognize_digits(digits)

print("Recognized digits:", recognized_digits)