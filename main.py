import cv2
import numpy as np
from scipy.fftpack import dct, idct
import pywt # PyWavelets Python_ >=3.7 NumPy>= 1.14.6 Cython >= 0.23.5 Matplotlib SciPy DWT 이산웨이블릿 변환
#pip install pywavelets
np.seterr(divide='ignore', invalid='ignore') # RuntimeWarning: invalid value encountered in divide오류 해결
def main():
    input = cv2.imread("figures/woman.jpg")
    cv2.imshow("input", input)
    #print(input.shape)# height x width x channel 행1080 열1440 채널3
    output = reflectSuppress(input, 0.125, 1e-8) #thresholding을 0~0.150변경해서 reflection삭제 강도 정함
    output1 = reflectSuppress(input, 0.150, 1e-8)
    # test(input, 0.033, 1e-8)
    Ref = cv2.subtract( input, output, dtype=cv2.CV_64F)
    cv2.imshow("Output_threshold-0.125", output)
    cv2.imshow("Output1_threshold_0.150", output1)
    cv2.imshow("reflect1",cv2.absdiff(input.astype(np.int8), output.astype(np.int8)))
    cv2.imshow("reflect2", input.astype(np.int8)  - output.astype(np.int8))

    #cv2.absdiff(input.astype(np.float64), output.astype(np.float64)) 완전 실패
    #cv2.subtract(input, output, dtype=cv2.CV_64F) 완전 실패

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def matlab_mat2grey(A):
    alpha = min(A.flatten())
    beta = max(A.flatten())
    I = A
    cv2.normalize(A, I, alpha , beta ,cv2.NORM_MINMAX)
    I = np.uint8(I)
    return I

# 매트랩에서 이미지  => 예를 들어, 픽셀 (10,5)의 빨간색, 녹색 및 파란색 색 구성요소는 각각 RGB(10,5,1), RGB(10,5,2) 및 RGB(10,5,3)에 저장

"""파이썬
# RGB 색 평면 분할
b_plane, g_plane, r_plane = cv2.split(src)


# 슬라이싱 이용하여 RGB 색 평면 분할
b_plane = src[:, :, 0]
g_plane = src[:, :, 1]
r_plane = src[:, :, 2]
"""
#h, w, c = im.shape = height X width X
#인풋이미지, gradient Thresholding parameter 0 to 0.150,
def reflectSuppress(lm, h, epsilon):
    Y = im2double(lm)
    [m, n, r] = np.shape(Y)
    T = np.zeros((m, n, r))
    Y_Laplacian_2 = np.zeros((m, n, r))
    for dim in range(0,r):
        GRAD = grad(Y[:,:, dim])
        GRAD_x = GRAD[:,:, 0]
        GRAD_y = GRAD[:,:, 1]
        GRAD_norm = np.sqrt((GRAD_x ** 2) + (GRAD_y ** 2))
        #GRAD_norm_thresh = pywt.threshold(GRAD_norm, h, 'hard')
        GRAD_norm_thresh = wthresh(GRAD_norm, 'h', h)
        for i in range(len(GRAD_norm_thresh)):
            for j in range(len(GRAD_norm_thresh[i])):
                if GRAD_norm_thresh[i][j] == 0:
                    GRAD_x[i][j] = 0
        for i in range(len(GRAD_norm_thresh)):
            for j in range(len(GRAD_norm_thresh[i])):
                if GRAD_norm_thresh[i][j] == 0:
                    GRAD_y[i][j] = 0
        GRAD_thresh = np.zeros((m, n, 3))
        GRAD_thresh[:,:, 0] = GRAD_x
        GRAD_thresh[:,:, 1] = GRAD_y
        Y_Laplacian_2[:,:, dim] = div(grad(div(GRAD_thresh))) # compute L(div(delta_h(Y)))
    rhs = Y_Laplacian_2 + epsilon * Y
    for dim in range (0,r):
        T[:,:, dim] = PoissonDCT_variant(rhs[:,:, dim], 1, 0, epsilon)
    return T

def wthresh(X,SORH,T): # good
    """
    doing either hard (if SORH = 'h') or soft (if SORH = 's') thresholding

    Parameters
    ----------
    X: array
         input data (vector or matrix)
    SORH: str
         's': soft thresholding
         'h' : hard thresholding
    T: float
          threshold value

    Returns
    -------
    Y: array_like
         output

    Examples
    --------
    y = np.linspace(-1,1,100)
    thr = 0.4
    ythard = wthresh(y,'h',thr)
    ytsoft = wthresh(y,'s',thr)
    """
    if ((SORH) != 'h' and (SORH) != 's'):
        print(' SORH must be either h or s')

    elif (SORH == 'h'):
        Y = X * (np.abs(X) > T)
        return Y
    elif (SORH == 's'):
        res = (np.abs(X) - T)
        res = (res + np.abs(res))/2.
        Y = np.sign(X)*res
        return Y

# solve the equation  (mu*L^2 - lambda*L + epsilon)*u = rhs via DCT
# where L means Laplacian operator
def PoissonDCT_variant(rhs, mu, Lambda, epsilon):
    [M, N] = np.shape(rhs)
    k = np.arange(1, M + 1)
    k = k.reshape(1, M)
    l = np.arange(1, N + 1)
    l = l.reshape(1, N)
    k = k.conjugate()
    k = np.transpose(k)
    eN = np.ones((1, N))
    eM = np.ones((M, 1))
    k = np.cos(np.pi / M * (k - 1))
    l = np.cos(np.pi / N * (l - 1))
    k = np.kron(k, eN)
    l = np.kron(eM, l)
    kappa = 2 * (k + l - 2)
    const = mu * (kappa**2) - Lambda *kappa + epsilon
    u = dct2(rhs)
    u = u/const
    u = idct2(u)
    return u

def dct2(a): #2차원 이산 코사인 변환
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):#2-D inverse discrete cosine transform
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

def grad(A):   # compute the gradient of a 2D image array
        [m,n]=np.shape(A)
        B = np.zeros((m, n, 2))
        Ar = np.zeros((m, n))
        Ar[:, 0: n - 1]=A[:, 1: n]
        Ar[:,n-1] = A[:,n-1]

        Au = np.zeros((m, n))
        Au[0: m - 1 ,:]=A[1: m,:]
        Au[m-1,:]=A[m-1,:]
        B[:,:, 0]=Ar - A
        B[:,:, 1]=Au - A
        return B

"""
compute the divergence of gradient
Input A is a matrix of size m*n*2
A(:,:,1) is the derivative along the x direction
A(:,:,2) is the derivative along the y direction
"""
def div(A):
        [m, n, r] = np.shape(A)
        B = np.zeros((m, n))
        # the derivative along the x direction
        T = A[:, :, 0]
        T1 = np.zeros((m, n))
        T1[:, 1: n]=T[:, 0: n - 1]
        B = B + T - T1
        #derivative along the y direction
        T = A[:, :, 1]
        T1 = np.zeros((m, n))
        T1[1: m,:]=T[0: m - 1,:]
        B = B + T - T1
        return B

def im2double(im):#good
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # #good

if __name__ == "__main__":
	main()