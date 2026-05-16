Input 64x64x3

Stem:
Conv(3→32, 3x3)
BN
ReLU

Block 1:
Residual block 32→32
Residual block 32→32
MaxPool

Block 2:
Residual block 32→64
Residual block 64→64
MaxPool

Block 3:
Residual block 64→128
Residual block 128→128
GlobalAvgPool

MLP:
Linear 128→64
Dropout
ReLU
Linear 64→1