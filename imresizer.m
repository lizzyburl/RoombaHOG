path = 'C:/Users/Lizzy/Documents/ComputerVision/FinalProject/notPedestrians128x64/'

for i = 1:1:270
    filename = sprintf('no_person__no_bike_%03d.bmp', i);
    im = imread(filename);
    imCorrectAspect = im(1:480, 100:340, :);
    imToSave = imresize(imCorrectAspect,  [128 64]);
    imwrite(imToSave, filename);
end
