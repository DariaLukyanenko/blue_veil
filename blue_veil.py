import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle

# считаем rgb по вуали
def calculate_average_rgb_veil(image_array, threshold=10):
    # Используйте маску, чтобы определить область без черного цвета
    non_black_mask = np.all(image_array > threshold, axis=-1)

    # Примените маску к каналам RGB
    non_black_red = image_array[:,:,0][non_black_mask]
    non_black_green = image_array[:,:,1][non_black_mask]
    non_black_blue = image_array[:,:,2][non_black_mask]

    # Рассчет среднего значения для каждого канала
    average_red = np.mean(non_black_red)
    average_green = np.mean(non_black_green)
    average_blue = np.mean(non_black_blue)

    # Рассчет среднего отклонения для каждого канала
    std_red = np.std(non_black_red)
    std_green = np.std(non_black_green)
    std_blue = np.std(non_black_blue)

    return average_red, average_green, average_blue, std_red, std_green, std_blue

# считаем hsv по вуали
def calculate_average_hsv_veil(image_array, threshold=10):
    # Преобразование RGB в HSV
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Используйте маску, чтобы определить область без черного цвета
    non_black_mask = np.all(image_array > threshold, axis=-1)

    # Примените маску к каналам HSV
    non_black_hue = hsv_image[:, :, 0][non_black_mask]
    non_black_saturation = hsv_image[:, :, 1][non_black_mask]
    non_black_value = hsv_image[:, :, 2][non_black_mask]

    # Рассчет среднего значения для каждого канала
    average_hue = np.mean(non_black_hue)
    average_saturation = np.mean(non_black_saturation)
    average_value = np.mean(non_black_value)

    # Рассчет среднего отклонения для каждого канала
    std_hue = np.std(non_black_hue)
    std_saturation = np.std(non_black_saturation)
    std_value = np.std(non_black_value)

    return average_hue, average_saturation, average_value, std_hue, std_saturation, std_value


def calculate_non_black_area(veil_img, mel_img):
    image1 = veil_img
    image2 = mel_img
    # Преобразование изображения в оттенки серого
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Бинаризация изображения
    _, binary_image1 = cv2.threshold(gray_image1, 1, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(gray_image2, 1, 255, cv2.THRESH_BINARY)
    # Вычисление площади области, где нет черного цвета
    non_black_area1 = cv2.countNonZero(binary_image1)
    non_black_area2 = cv2.countNonZero(binary_image2)
    percentage = non_black_area1/non_black_area2
    return percentage


def process_image(image_path):
    def do_nothing():
        pass

    # create slider here
    cv2.namedWindow("Slider")
    cv2.resizeWindow("Slider", 400, 300)
    cv2.createTrackbar("Hue Min", "Slider", 0, 179, do_nothing)  # Hue range is 0-179 in OpenCV
    cv2.createTrackbar("Hue Max", "Slider", 179, 179, do_nothing)
    cv2.createTrackbar("Saturation Min", "Slider", 0, 255, do_nothing)
    cv2.createTrackbar("Saturation Max", "Slider", 255, 255, do_nothing)
    cv2.createTrackbar("Value Min", "Slider", 0, 255, do_nothing)
    cv2.createTrackbar("Value Max", "Slider", 255, 255, do_nothing)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 300))
    # convert to HSV image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        hue_min = cv2.getTrackbarPos("Hue Min", "Slider")
        hue_max = cv2.getTrackbarPos("Hue Max", "Slider")
        sat_min = cv2.getTrackbarPos("Saturation Min", "Slider")
        sat_max = cv2.getTrackbarPos("Saturation Max", "Slider")
        val_min = cv2.getTrackbarPos("Value Min", "Slider")
        val_max = cv2.getTrackbarPos("Value Max", "Slider")

        # set bounds
        lower_bound = np.array([hue_min, sat_min, val_min])
        upper_bound = np.array([hue_max, sat_max, val_max])

        # create mask
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        resulting_img = cv2.bitwise_and(img, img, mask=mask)

        stacked_imgs = np.hstack([img, resulting_img])

        # create a stacked image of the original and the HSV one.
        cv2.imshow("Image", stacked_imgs)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Save processed image to output folder
            processed_image_path = f"processed_{image_path}"
            cv2.imwrite(processed_image_path, resulting_img)
            print("Processed image saved as:", processed_image_path)
            break

    cv2.destroyAllWindows()
    return resulting_img


def count_characteristics(image_path):
    characters = {}

    veil_img = process_image(image_path)
    veil_avg_red, veil_avg_green, veil_avg_blue, veil_std_red, veil_std_green, veil_std_blue = calculate_average_rgb_veil(veil_img)
    veil_avg_hue, veil_avg_value, veil_avg_saturation, veil_std_hue, veil_std_value, veil_std_saturation = calculate_average_hsv_veil(veil_img)

    mel_img = process_image(image_path)
    mel_avg_red, mel_avg_green, mel_avg_blue, mel_std_red, mel_std_green, mel_std_blue = calculate_average_rgb_veil(mel_img)
    mel_avg_hue, mel_avg_value, mel_avg_saturation, mel_std_hue, mel_std_value, mel_std_saturation = calculate_average_hsv_veil(mel_img)

    percentage = calculate_non_black_area(veil_img, mel_img)

    characters['Percentage'] = percentage

    characters['Average_Red'] = veil_avg_red
    characters['Average_Green'] = veil_avg_green
    characters['Average_Blue'] = veil_avg_blue

    characters['Std_Red'] = veil_std_red
    characters['Std_Green'] = veil_std_green
    characters['Std_Blue'] = veil_std_blue

    characters['Average_Hue'] = veil_avg_hue
    characters['Average_Saturation'] = veil_avg_value
    characters['Average_Value'] = veil_avg_saturation

    characters['Std_Hue'] = veil_std_hue
    characters['Std_Saturation'] = veil_std_value
    characters['Std_Value'] = veil_std_saturation

    # characters['mel_avg_red'] = mel_avg_red
    # characters['mel_avg_green'] = mel_avg_green
    # characters['mel_avg_blue'] = mel_avg_blue
    #
    # characters['mel_std_red'] = mel_std_red
    # characters['mel_std_green'] = mel_std_green
    # characters['mel_std_blue'] = mel_std_blue
    #
    # characters['mel_avg_hue'] = mel_avg_hue
    # characters['mel_avg_value'] = mel_avg_value
    # characters['mel_avg_saturation'] = mel_avg_saturation
    #
    # characters['mel_std_hue'] = mel_std_hue
    # characters['mel_std_value'] = mel_std_value
    # characters['mel_std_saturation'] = mel_std_saturation

    return characters


def create_df():
  full_df = pd.DataFrame()

  file_path = 'nevus'
  for num, filename in enumerate(os.listdir(file_path)):
      info_img = {}
      file_path = 'nevus'
      img_path = os.path.join(file_path, filename)
      # img = cv2.imread(img_path)
      # img, seg_img = seg_area_of_interest(img)
      characteristics_of_img: dict = count_characteristics(img_path)
      info_img['image'] = filename
      info_img.update(characteristics_of_img)
      info_img['label'] = 'невус'
      df = pd.DataFrame(info_img, index=[num])
      full_df = pd.concat([full_df, df])

  file_path = 'djest1'
  for num, filename in enumerate(os.listdir(file_path)):
      info_img = {}
      file_path = 'djest1'
      img_path = os.path.join(file_path, filename)
      # img = cv2.imread(img_path)
      # img, seg_img = seg_area_of_interest(img)
      characteristics_of_img: dict = count_characteristics(img_path)
      info_img['image'] = filename
      info_img.update(characteristics_of_img)
      info_img['label'] = 'меланома'
      df = pd.DataFrame(info_img, index=[num])
      full_df = pd.concat([full_df, df])

  return full_df


def create_model():
    df = create_df()

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['Filename', 'Diagnosis'], axis=1),
        df['Diagnosis'],
        test_size=0.2,
        random_state=42,
        shuffle=True)

    clf_1 = RandomForestClassifier(random_state=42)
    clf_2 = GaussianNB()
    clf_3 = svm.SVC(probability=True, random_state=42)

    eclf_stack = StackingClassifier(
        estimators=[('rfc', clf_1), ('gnb', clf_2)],
        final_estimator=svm.SVC(probability=True, random_state=42))

    eclf_stack = eclf_stack.fit(X_train, y_train)
    y_pred3 = eclf_stack.predict(X_test)

    with open('eclf_stack.pkl', 'wb') as file:
        pickle.dump(eclf_stack, file)

    print(classification_report(y_test, y_pred3))
    return eclf_stack


# def slow_pred(image_path) -> str:
#     clf = create_model()
#     info_img = {}
#     set_ch: dict = count_characteristics(image_path)
#     info_img.update(set_ch)
#     df = pd.DataFrame(info_img, index=[0])
#     res = clf.predict(df)
#     if res == 1:
#         return 'nevus'
#     if res == 2:
#         return 'melanoma'

def first_pred(image_path) -> str:
    path = 'merged_file.csv'
    df = pd.read_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['Filename', 'Diagnosis'], axis=1),
        df['Diagnosis'],
        test_size=0.2,
        random_state=42,
        shuffle=True)
    clf_1 = RandomForestClassifier(random_state=42)
    clf_2 = GaussianNB()
    clf_3 = svm.SVC(probability=True, random_state=42)

    eclf_stack = StackingClassifier(
        estimators=[('rfc', clf_1), ('gnb', clf_2)],
        final_estimator=svm.SVC(probability=True, random_state=42))

    clf = eclf_stack.fit(X_train, y_train)
    y_pred3 = eclf_stack.predict(X_test)

    with open('clf.pkl', 'wb') as file:
        pickle.dump(eclf_stack, file)

    print(classification_report(y_test, y_pred3))

    info_img = {}
    set_ch: dict = count_characteristics(image_path)
    info_img.update(set_ch)
    df = pd.DataFrame(info_img, index=[0])
    res = clf.predict(df)
    return res


def fast_pred(image_path) -> str:
    with open('clf.pkl', 'rb') as file:
        clf = pickle.load(file)
    info_img = {}
    set_ch: dict = count_characteristics(image_path)
    info_img.update(set_ch)
    df = pd.DataFrame(info_img, index=[0])
    res = clf.predict(df)
    return res

res = fast_pred('djest1\\28.jpg')
print(res)
