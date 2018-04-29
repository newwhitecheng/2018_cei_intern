function res = imagenet_eval(path, comp_id, test_set, output_dir, rm_res)

synsets = get_imagenet_opts(path);

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


%for i = 1:200
%  cls = imagenet_opts(i).name;
%  res(i) = imagenet_eval_cls(cls, imagenet_opts, comp_id, output_dir, rm_res, test_set, path);
%end

%fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
%fprintf('Results:\n');
%aps = [res(:).ap]';
%fprintf('%.1f\n', aps * 100);
%fprintf('%.1f\n', mean(aps) * 100);
%fprintf('~~~~~~~~~~~~~~~~~~~~\n');


function res = imagenet_eval_cls(cls, imagenet_opts, comp_id, output_dir, rm_res, test_set, path)

addpath(fullfile(path, 'evaluation'));

res_fn = sprintf(path, '/results/', [test_set, '/'], comp_id, cls);

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

do_eval = (str2num(year) <= 2007) | ~strcmp(test_set, 'test');
if do_eval
  % Bug in VOCevaldet requires that tic has been called first
  tic;
  [recall, prec, ap] = VOCevaldet(VOCopts, comp_id, cls, true);
  ap_auc = xVOCap(recall, prec);

  % force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', ...
        [output_dir '/' cls '_pr.jpg']);
end
fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

save([output_dir '/' cls '_pr.mat'], ...
     'res', 'recall', 'prec', 'ap', 'ap_auc');

if rm_res
  delete(res_fn);
end

rmpath(fullfile(VOCopts.datadir, 'VOCcode'));
