import cv2
import os

if __name__ == "__main__":
    path = '../cut_img/8/'
    path_new = '../new_cut/8'
    imgs = os.listdir(path)
    print(imgs)
    for img in imgs:
        #Загружаем изображение
        image = cv2.imread(os.path.join(path, img))
        crop = image[30:270, 30:270]
        crop_gauss = cv2.GaussianBlur(crop, (9, 9), 10)
        low_border = (84, 84, 84)
        high_border = (240, 240, 240)
        filt = cv2.inRange(crop_gauss, low_border, high_border)
        contour, hierarchy = cv2.findContours(filt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_coord = []
        y_coord = []
        for i in contour:
            for j in i:
                for xy in j:
                    x_coord.append(xy[0])
                    y_coord.append(xy[1])
        crop1 = crop[min(y_coord):max(y_coord), min(x_coord):max(x_coord)]
        #resize = cv2.resize(crop1, (32, 32))
        #cv2.imshow('test', crop1)
        cv2.imwrite(os.path.join(path_new, img), crop1)
        cv2.waitKey(0)
