from PIL import Image
import face_recognition
import numpy
from itertools import islice

# Load the jpg file into a numpy array
index_txt=open("./ValidationList.txt")
start = 0 * 4314
count = 0
count2 = 0
result = numpy.empty(shape=[0, 128], dtype='f')

for f in islice(index_txt, start, start + 50):
    #print(start)
    f = f.strip('\n')
    count += 1
    print("Processing {} file: {}".format(count, f))
    image = face_recognition.load_image_file(f)

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image, model='cnn')

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    if len(face_locations) == 0:
        count2 += 1

    print((face_recognition.face_encodings(image)[0]).dtype)
    #result = face_recognition.face_encodings(f, face_locations)

print("Failed result: {}".format(count2))
# for face_location in face_locations:
#
#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#
#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()