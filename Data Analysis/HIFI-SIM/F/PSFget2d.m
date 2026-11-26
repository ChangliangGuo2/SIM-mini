function  [output] = PSFget2d(sigma)
psfsigma = sigma;
psfN = ceil(psfsigma./sqrt(8*log(2)) * sqrt(-2 * log(0.0002))) + 1;
psfN = psfN * 2 + 1;
output = Gauss(psfsigma,psfN);
end