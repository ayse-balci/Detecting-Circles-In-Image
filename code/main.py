import cv2
import numpy as np
import math


def CreateAccumulatorList(image, height, width, radius_min, radius_max):
    accumulator = np.zeros((radius_max, height, width))
    for r in range(radius_min, radius_max):
        for m in range(height):
            for n in range(width):
                if image[m][n] == 255:
                    for theta in range(0, 360, 10):
                        x0 = int(m - r * math.cos(theta * math.pi / 180))
                        y0 = int(n - r * math.sin(theta * math.pi / 180))
                        if x0 > 0 and x0 < height and y0 > 0 and y0 < width:
                            accumulator[r][x0][y0] += 1
    return accumulator


def DetectHoughCircles(image, accumulator, height, width, radius_min, radius_max, threshold):

    slide_pixel_amount = 30
    max_values = np.zeros((radius_max, slide_pixel_amount, slide_pixel_amount))
    max_value = np.amax(accumulator)

    final_circles = []
    for r in range(radius_min, radius_max):
        accumulator[r] = accumulator[r] / max_value  # Normalize values

    for i in range(0, height - slide_pixel_amount, slide_pixel_amount):
        for j in range(0, width - slide_pixel_amount, slide_pixel_amount):
            max_values = accumulator[:, i:i + slide_pixel_amount, j:j + slide_pixel_amount]

            max_val = np.where(max_values == max_values.max())
            r, x, y = max_val[0][0], max_val[1][0], max_val[2][0]

            if max_values.max() > threshold:
                accumulator[:, i:i + slide_pixel_amount, j:j + slide_pixel_amount] = accumulator[:, i:i + slide_pixel_amount, j:j + slide_pixel_amount] / \
                                                           accumulator[r][x + i][
                                                               y + j]
                for r in range(radius_min, radius_max):
                    for q in range(i, i + slide_pixel_amount):
                        for w in range(j, j + slide_pixel_amount):
                            if accumulator[r][q][w] > 0.9:
                                cv2.circle(image, (w, q), r, (0, 255, 0), 2)
                                final_circles.append((float(w), float(q), float(r)))
                                #print(w, q, r)
    return image, final_circles


def CannyEdgeDetection(filename):
    # Read the input image
    inputImage = cv2.imread(filename, 1)
    cv2.imshow('Input Image', inputImage)
    cv2.waitKey(0)
    # Gaussian Blurring of Gray Image to reduce noise
    gaussImage = cv2.GaussianBlur(inputImage, (3, 3), 0)
    cv2.imshow('Gaussian Blurring of Gray Image', gaussImage)
    cv2.waitKey(0)
    # Using OpenCV Canny Edge detector to detect edges
    cannyImage = cv2.Canny(gaussImage, 75, 150)
    cv2.imshow('Canny Edge Detected Image', cannyImage)
    cv2.waitKey(0)

    return inputImage, cannyImage


def main():

    filename_list = []

    for i in range(3, 114):
        filename = 'dataset/Images/' + str(i) + '.jpg'
        filename_list.append((filename, str(i)))

    threshold = 0.6

    for filename, file in filename_list:
        print(file,'.jpg Hough Circle Detection is starting' )
        inputImage, cannyImage = CannyEdgeDetection(filename)
        height = inputImage.shape[0]
        width = inputImage.shape[1]
        radius_min = 10
        radius_max = 200

        accumulator = CreateAccumulatorList(cannyImage, height, width, radius_min, radius_max)

        output_image, final_circles = DetectHoughCircles(inputImage, accumulator, height, width, radius_min, radius_max, threshold)

        file = open('dataset/GT/' + str(file) + '.txt' , 'r')
        lines = file.readlines()
        lines.pop(0)

        iou = 0
        count = 0
        findedCircleCount = 0
        couldNotFindCircleCount = 0

        for line in lines:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            r = float(line.strip().split(' ')[2])
            mainCircleArea = math.pi * r * r
            list = []
            for detected_x, detected_y, detected_r in final_circles:

                distanceBetweenCenters = math.sqrt(abs(x - detected_x) ** 2 + (abs(y - detected_y)) ** 2)

                if (distanceBetweenCenters < 3 and abs(r - detected_r) < 3):
                    findedCircleCount += 1
                    list.append((detected_x, detected_y, detected_r))

            if len(list) > 0:
                print('for ', str(x), ' ' , str(y), ' ', str(r), ' detected circles are:')
                for detected_x, detected_y, detected_r in list:
                    intersection = areaOfIntersection(x, y, r, detected_x, detected_y, detected_r)
                    detectedCircleArea = math.pi * detected_r * detected_r
                    iou_line = intersection / (mainCircleArea + detectedCircleArea - intersection)
                    iou += iou_line
                    print(detected_x, ' ', detected_y, ' ', detected_r , ' detected line Intersection over Union (IoU) score is iou ', iou_line)
                    count += 1
            else:
                print('for ', line, ' no circle detected')
                couldNotFindCircleCount += 1
                count += 1

        print(findedCircleCount, ' circle of original image is founded')
        print(couldNotFindCircleCount, ' circle of original image is not founded')
        print('Intersection over Union (IoU) score in result is ', iou / count)

        cv2.imshow('Circle Detected Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def areaOfIntersection(x0, y0, r0, x1, y1, r1):
    distance = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    if (distance > r1 + r0):
        return 0

    elif (distance <= abs(r0 - r1) and r0 >= r1):
        # Return area of circle1
        return math.pi * r1 * r1

    elif (distance <= abs(r0 - r1) and r0 < r1):
        # Return area of circle0
        return math.pi * r0 * r0

    else:
        phi = (math.acos((r0 * r0 + (distance * distance) - r1 * r1) / (2 * r0 * distance))) * 2
        theta = (math.acos((r1 * r1 + (distance * distance) - r0 * r0) / (2 * r1 * distance))) * 2
        area1 = 0.5 * theta * r1 * r1 - 0.5 * r1 * r1 * math.sin(theta)
        area2 = 0.5 * phi * r0 * r0 - 0.5 * r0 * r0 * math.sin(phi)

    # Return area of intersection
    return area1 + area2


if __name__ == "__main__":
    main()
