import cv2
import imutils
import numpy as np
import os
from scipy import ndimage

IMG_DIR = "img-dir"
TMPL_DIR = "tmpl-dir"
RES_DIR = "res-dir"


def make_templates():
    """ Makes rotation of 360 degrees from base template in TMPL_DIR.
    Saves rotated templates as tmpl{deg}.png in TMPL_DIR
    """
    try:
        base = cv2.imread(os.path.join(TMPL_DIR, 'minion-template.png'))
    except IOError:
        print('Failed to make templates. Base template is not found')
        return
    for deg in range(360):
        tmpl = ndimage.rotate(base, deg)
        cv2.imwrite(os.path.join(TMPL_DIR, 'tmpl%d.png' % deg), tmpl)
    return


if __name__ == '__main__':
    # Open the main image and convert it to gray scale image
    main_image = cv2.imread(os.path.join(IMG_DIR, "minions.png"))
    gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    threshold = 0.8
    make_templates()

    # Compare every template with different degree of rotation with the objects in the image
    for template in os.listdir(TMPL_DIR):
        templ = cv2.imread(os.path.join(TMPL_DIR, template), 0)

        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(templ, width=int(templ.shape[1] * scale))
            width, height = resized.shape[::-1]

            # Match the template using cv2.matchTemplate
            res = cv2.matchTemplate(gray_image, resized, cv2.TM_CCOEFF_NORMED)

            # Get the location of template in the image
            loc = np.where(res >= threshold)

            for point in zip(*loc[::-1]):
                # Draw the rectangle around the matched template
                cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 0, 255))
                cv2.imwrite(os.path.join(RES_DIR, "minion-res.png"), main_image)
    print("Check the results in res-dir directory!")
