from PIL import Image, ImageDraw

# Load the image
image_path = '/home/jetson/llamaR/Llama3-Playground/llava/planning/404.png'
image = Image.open(image_path)

# Define the coordinates
width, height = image.size
coords = [0.18 * width, 0.32 * height, 0.45 * width, 0.65 * height]

# Draw the box
draw = ImageDraw.Draw(image)
draw.rectangle(coords, outline="red", width=3)

# Save the modified image
output_path = "/mnt/data/dog_bike_car_boxed.jpg"
image.save(output_path)

output_path

# Load the new image
image_path_2 = "/mnt/data/404.png"
image_2 = Image.open(image_path_2)

# Define the new coordinates
width_2, height_2 = image_2.size
coords_2 = [0.487 * width_2, 0.392 * height_2, 0.659 * width_2, 0.641 * height_2]

# Draw the box
draw_2 = ImageDraw.Draw(image_2)
draw_2.rectangle(coords_2, outline="red", width=3)

# Save the modified image
output_path_2 = "/mnt/data/404_boxed.png"
image_2.save(output_path_2)

output_path_2
