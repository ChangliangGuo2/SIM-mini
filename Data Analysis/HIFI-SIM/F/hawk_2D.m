function [output] = hawk_2D(input)
addpath('F');
cdfile = cd;
javaaddpath ([cdfile,'\F\MIJscripts\mij.jar']);
javaaddpath ([cdfile,'\F\MIJscripts\ij.jar']);
addpath ([cdfile,'\F\MIJscripts\scripts']);


Miji;
MIJ.createImage(squeeze(input)); 
MIJ.run("HAWK", "number=3 negative=Separate output=[Group temporally]");
output = MIJ.getCurrentImage();
MIJ.run('Close All');
MIJ.closeAllWindows;
MIJ.exit;
