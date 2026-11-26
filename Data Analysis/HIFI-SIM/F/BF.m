function  [output] = BF(input,backg)
backgrounds = background_estimation(input./backg,1,7,'db6',3);
backgrounds(backgrounds < 0) = 0;
output = input - backgrounds;
output(output < 0) = 0;
end