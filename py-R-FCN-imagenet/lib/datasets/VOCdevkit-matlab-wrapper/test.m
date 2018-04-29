path='/home/hsc38/Data/ImageNet/ILSVRC13';
synsets = get_imagenet_opts(path);
comp_id = '15523';
test_set = 'DET_val';
addpath(fullfile(path, 'evaluation'));
pred_file= [path, '/results/', comp_id, '_det_', test_set, '.txt'];
meta_file = [path, '/data/meta_det.mat'];
eval_file = [path, '/data/det_lists/', 'val.txt'];
blacklist_file = [path, '/data/ILSVRC2013_det_validation_blacklist.txt'];

optional_cache_file = '';
gtruth_directory = [path, '/ILSVRC2013_bbox_DET_val'];

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);
fprintf('gtruth_directory: %s\n', gtruth_directory);
if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

while isempty(gtruth_directory)
    g_dir = input(['Please enter the path to the Validation bounding box ' ...
                   'annotations directory: '],'s');
    d = dir(sprintf('%s/*val*.xml',g_dir));
    if length(d) == 0
        fprintf(['does not seem to be the correct directory, please ' ...
                 'try again\n']);
    else
        gtruth_directory = g_dir;
    end
end

[ap recall precision] = eval_detection(pred_file,gtruth_directory,meta_file,eval_file,blacklist_file,optional_cache_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i=[1:200]
    s = synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
end
fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));

