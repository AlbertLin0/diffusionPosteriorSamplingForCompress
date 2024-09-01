import numpy as np
def r_channel_histogram(img):
    flatten_r_channel_array = np.sort(img[0][0].flatten()) 
    length = len(flatten_r_channel_array)
    pdf = []
    bin_size = 2.0 / 256.0 
    for i in range(256):
        density = 0
        lower_bound = i * bin_size - 1
        upper_bound = (i + 1) * bin_size - 1

        for j in range(length):
            if flatten_r_channel_array[j] >= lower_bound and flatten_r_channel_array[j] < upper_bound:
                density += 1
            elif flatten_r_channel_array[j] < lower_bound:
                continue
            else:
                pdf.append(float(density) / length)
                break
    return pdf 




if __name__ == "__main__":
    pass