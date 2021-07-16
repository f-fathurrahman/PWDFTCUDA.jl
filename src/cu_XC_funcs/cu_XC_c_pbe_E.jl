# epsxc only version
function cu_XC_c_pbe_E( rho, grho )

    ga = 0.0310906908696548950
    be = 0.06672455060314922
    third = 1.0/3.0
    pi34 = 0.6203504908994
    xkf = 1.919158292677513
    xks = 1.128379167095513

    rs = pi34/CUDA.pow(rho, third)
    ec = cu_XC_c_pw_E( rho )
    
    kf = xkf/rs
    ks = xks * CUDA.sqrt(kf)
    t = CUDA.sqrt(grho) / (2.0 * ks * rho)
    
    expe = CUDA.exp(-ec/ga)
    af = be / ga * (1.0 / (expe - 1.0) )  
  
    y = af * t * t
  
    xy = (1.0 + y) / (1.0 + y + y * y)
  
    qy = y * y * (2.0 + y) / (1.0 + y + y * y)^2
  
    s1 = 1.0 + be / ga * t * t * xy
    h0 = ga * CUDA.log(s1)
  
    sc = rho * h0

    return sc

end