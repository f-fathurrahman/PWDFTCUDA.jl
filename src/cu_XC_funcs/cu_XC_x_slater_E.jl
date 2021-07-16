# epxsc only version
function cu_XC_x_slater_E( Rhoe )

    third = 1.0/3.0
    pi34 = 0.6203504908994  # pi34=(3/4pi)^(1/3)
    rs = pi34/CUDA.pow(Rhoe, third)

    f = -0.687247939924714
    alpha = 2.0/3.0

    ex = f * alpha / rs
    return ex
end