clc;
clear;
close all;

filename = input('Please enter the filename: ','s');

% Paths relative to project root (run from project root)
frame = imread(['data/mot20/frame/',filename,'.png']);
[height,width,~] = size(frame);
area_path = ['data/mot20/area/',filename,'.mat'];
if exist(area_path,'file')
    disp('The walkable area has been loaded.');
    area = load(area_path).area;
else
    area = ones(height,width);
end

figure;
imshow(frame,'InitialMagnification','fit');
hold on;
title('Walkable Area');
blue = cat(3,zeros(height,width),zeros(height,width),ones(height,width));
mask = imshow(blue);
set(mask,'AlphaData',0.5*area);

while true
    state = input(['Please enter the operation type: ' ...
        '1 - Remove Area, 2 - Add Area. ' ...
        'Press 0 to exit.']);
    if state == 0
        break;
    end
    switch state
        case 1
            disp('Please draw the removed polygon.')
            polygon = drawpolygon('LineWidth',2,'Color','r');
            position = round(polygon.Position);
            removed_area = poly2mask(position(:,1),position(:,2),height,width);
            area = area&~removed_area;
        case 2
            disp('Please draw the added polygon.')
            polygon = drawpolygon('LineWidth',2,'Color','g');
            position = round(polygon.Position);
            added_area = poly2mask(position(:,1),position(:,2),height,width);
            area = area|added_area;
    end
    cla;
    imshow(frame);
    hold on;
    blue = cat(3,zeros(height,width),zeros(height,width),ones(height,width));
    mask = imshow(blue);
    set(mask,'AlphaData',0.5*area);
end

area_dir = ['data/mot20/area/'];
if ~exist(area_dir,'dir')
    mkdir(area_dir);
end
save(area_path,'area');
disp(['The walkable area has been saved as ',filename,'.mat']);
