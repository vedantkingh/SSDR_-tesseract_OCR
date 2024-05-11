import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # # Plot histogram
    # plt.plot(hist, color='black')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.title('Grayscale Image Histogram')
    # plt.show()

    blur_image = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2)
    filename = os.path.basename("thresh_seven_segment_display.png")
    cv2.imwrite(filename, thresh)
    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

# Load the image
image = cv2.imread("3.jpg")

# Preprocess the image
thresh = preprocess_image(image)

# Find contours
contours = find_contours(thresh)

# Filter contours
filtered_contours = filter_contours(contours)

# Extract digits
digits = extract_digits(image, filtered_contours)

# Recognize digits
recognized_digits = recognize_digits(digits)

# Print recognized digits
print("Recognized digits:", recognized_digits)