function [output] = hawk(input)
addpath('F');
cdfile = cd;
javaaddpath ([cdfile,'\F\MIJscripts\mij.jar']);
javaaddpath ([cdfile,'\F\MIJscripts\ij.jar']);
addpath ([cdfile,'\F\MIJscripts\scripts']);
Miji;
for i = 1:size(input,3)
MIJ.createImage(squeeze(input(:,:,i,:))); 
MIJ.run("HAWK", "number=3 negative=Separate output=[Group temporally]");
output(:,:,:,i) = MIJ.getCurrentImage();
MIJ.run('Close All');
end
MIJ.closeAllWindows;
MIJ.exit;
output = permute(output,[1,2,4,3]);
output = output./max(max(max(max(output))));
