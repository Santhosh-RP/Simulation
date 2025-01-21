import os
import random

total_images = 20000
images_per_chunk = 25
chunks = total_images // images_per_chunk
directory = 'images_randomised/'

chunk_list = [x for x in range(chunks)]

random.shuffle(chunk_list)


for chunk_dest, chunk_orig in enumerate(chunk_list):
    for image_in_chunk in range(images_per_chunk):
        orig = chunk_orig*images_per_chunk + image_in_chunk
        dest = chunk_dest*images_per_chunk + image_in_chunk
        orig_path = directory+f'image_{str(orig).zfill(6)}.png'
        dest_path = directory+f'image_shuffled_{str(dest).zfill(6)}_{str(orig).zfill(6)}.png'
        os.rename(orig_path, dest_path)

