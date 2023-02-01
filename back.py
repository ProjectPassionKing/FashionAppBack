from rembg import remove
from PIL import Image

input = Image.open("uploads\clothes\clothes_1.jpg")
output = remove(input)
output.save("uploads\clothes\clothes_1.jpg")
