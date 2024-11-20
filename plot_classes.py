import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter



"""
This script is just used to evaluate the mask dataset and determine (or verify) the number of classes
"""



def find_num_classes(mask_paths) -> dict:
    # Takes in an iterable containing the path to the masks, where it would look for unique pixel values
    # And returns the count, this would be the number of classes in the segmentation model

    print("Now calculating number of classes within the dataset...")

    # This varies based on resolution! Used to as filter to discard uncommon pixels which are likely artifacts
    # Wouldn't need it if mask is well-made
    # threshold = 100

    result = {}
    for i in range(len(mask_paths)):
        if i % 25 == 0:
            print(f"Now processing image index={i}")

        image = Image.open(mask_paths[i]).convert("RGB")
        # image.show()
        c = Counter(list(image.getdata()))  # Count the number of occurrences

        # Used to filter out artifacts, i.e. pixel value that rarely occurs
        # Only need it if there are aliasing/blend/artifacts
        # trunc = {}
        # for k, v in c.items():
        #     if v >= threshold:
        #         trunc[k] = v
        # result = {k: trunc.get(k, 0) + result.get(k, 0) for k in list(trunc.keys()) + list(result.keys())}

        result = {k: c.get(k, 0) + result.get(k, 0) for k in list(c.keys()) + list(result.keys())}

    return result



if __name__ == "__main__":
    mask_dir = r"./Dataset/ScaledMasks"  # Using Scaled Masks since Augmented have more "repetition"
    mask_paths = [os.path.join(mask_dir, path) for path in os.listdir(mask_dir)]

    # Wouldn't need to sample entire dataset, around ~100 should sample each class type at least once
    # Adjust if using a different dataset
    mask_paths = mask_paths[:100]


    result = find_num_classes(mask_paths)

    result = sorted(result.items(), key=lambda x: x[1], reverse=True)  # Sort based on values (The count)
    for i in range(len(result)):
        print(f"{i + 1}. Pixel {result[i][0]} occurred {result[i][1] // len(mask_paths)} times on average")


    pixel_values = [p[0] for p in result]
    r, g, b = [pv[0] for pv in pixel_values], [pv[1] for pv in pixel_values], [pv[2] for pv in pixel_values]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(r, g, b, c='blue', marker='o')

    # Labels
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")

    plt.show()
