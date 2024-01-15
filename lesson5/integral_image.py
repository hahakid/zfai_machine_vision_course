import numpy as np

# query the (i,j) of ii
def query_ii(ii, i, j, width, height):
    # D'= D-B-C+A
    return ii[j, i] - ii[j+height, i] - ii[j, i+width] + ii[j+height, i+width]

# build ii from input_im
def integralimage(input_im):
    '''
    in=[A, B    out=[A', B'
        C, D]        C', D']
    A'= A
    B'= A+B
    C'= A+C
    D'= D+B+C-A
    '''
    output_im = np.zeros_like(input_im, dtype=np.int32)
    #print(ii.shape)
    row, col = input_im.shape
    for i in range(row):
        for j in range(col):
            if i == 0 and j == 0:
                output_im[i, j] = input_im[i, j]  # A'= A
            elif i == 0 and j > 0:
                output_im[i, j] = input_im[i, j] + output_im[i, j - 1]  # B'= A + B
            elif j == 0 and i > 0:
                output_im[i, j] = input_im[i, j] + output_im[i-1, j]  # C'= A + C
            else:  # D' = A + B + C - D
                output_im[i, j] = input_im[i, j] + output_im[i-1, j] + output_im[i, j-1] - output_im[i-1, j-1]
    return output_im
