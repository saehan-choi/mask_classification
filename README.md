# efficientNet_B0 Analysis

EfficientNet(
  (conv_stem): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   
  512-3+2/2 + 1 ==> 256
  
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): SiLU(inplace=True)
  (blocks): Sequential(
    (0): Sequential(
      (0): DepthwiseSeparableConv(
        (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        256 ==> 256
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
				256 ==> 256
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
				256 ==> 256
          (gate): Sigmoid()
        )
        (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
				256 ==> 256
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): Identity()
      )
    )
    (1): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
				256 ==> 256
        (bn1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
				256-3+2/2+1 = 128
        (bn2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
				128 ==> 128
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
				128 ==> 128
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
				128 ==> 128
        (bn3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
				128 ==> 128
        (bn1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
				128-3+2/1 + 1 ==> 128
        (bn2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
				128 ==> 128
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
				128 ==> 128
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
				128 ==> 128
        (bn3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): Sequential(  
      (0): InvertedResidual(
        (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
				128 ==> 128
        (bn1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
				128-5+4/2+1= 64
        (bn2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64 
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
				64-5+4/1+1 = 64
        (bn2): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
				(64-3+2)/2+1 ==> 32
        (bn2): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
				64 ==> 64
          (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
				(64-3+2)/1+1 ==> 64
        (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): SiLU(inplace=True)
				64 ==> 64
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
				(64-3+2)/1 +1 ==> 64
        (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
				(64-5+4)/1+1 ==> 64
        (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
				(64-5+4)/1 + 1 ==> 64
        (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
				(64-5+4)/1 + 1 == 64
        (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
				64 ==> 64
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
				64 ==> 64
        (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True) 
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
				(32-5+4)/2+1 ==> 32
        (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
				32 ==> 32
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
				32 ==> 32
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
				32 ==> 32
        (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
				32 ==> 32
        (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)  
				(32-5+4)/2+1 ==> 16
        (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
				16 ==> 16
        (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
				16 ==> 16
        (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): InvertedResidual(
        (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): SiLU(inplace=True)
        (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
				16 ==> 16
        (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): SiLU(inplace=True)
        (se): SqueezeExcite(
          (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (act1): SiLU(inplace=True)
          (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
				16 ==> 16
          (gate): Sigmoid()
        )
        (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> 16
        (bn3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
				16 ==> (1280, 16, 16)
        
  (bn2): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): SiLU(inplace=True)
  (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
   AdaptivePooling을 하면 내가 원하는 kernel사이즈로 줄일 수 있습니다.

  (classifier): Linear(in_features=1280, out_features=1000, bias=True)
)
