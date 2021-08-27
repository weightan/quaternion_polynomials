
import numpy as np
import math

from tqdm import tqdm
from datetime import datetime

from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt


def log_dansity_map(val, max_count):

    brightness = math.log(val) / math.log(max_count)
    gamma = 2.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness


def display_with_plt(grid, N, spec):

    cmap = "hot"
    size = 10

    max_count = np.max(grid)

    for i in range(N):
        for j in range(N):
            if grid[i, j]:
                grid[i, j] = 256*log_dansity_map(grid[i, j], max_count)

    plt.figure(num = None, figsize=(size, size), dpi=300)

    plt.axis('off')

    plot = plt.imshow(grid, cmap = cmap )

    ####
    now =  str(datetime.now()).replace(" ", '_')
    now =  now.replace(":", '_')

    filenameImage = f'N_{N}_cmap_{cmap}_projection_{spec}_{now}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0)

    ####
    
    #plt.show()
    plt.close()


def display_with_pil(grid, N, spec, coeficients):
    cmap_name = "hot"
    cmap = cm.get_cmap(cmap_name)

    coef = [list(i) for i in coeficients]
    coef = [[round(j, 2) for j in i] for i in coeficients]

    aspect_ratio = 1
    width = N
    height = int(width * 1 / aspect_ratio)

    max_count = np.max(grid)
    print("Max count:", max_count)

    paint_zero_lvl = False
    mono_coloring = False

    im_arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if grid[y, x] != 0:
                if mono_coloring:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_dansity_map(grid[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])
            elif paint_zero_lvl:
                rgba = cmap( 0 )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    print(f"Saving image...{width}_{height}_spec_{spec}")

    now =  str(datetime.now()).replace(" ", '_')
    now =  now.replace(":", '_')

    im.save(f'N_{N}_cmap_{cmap_name}_{coef}_projection_{spec}_{now}.png')
    
    
