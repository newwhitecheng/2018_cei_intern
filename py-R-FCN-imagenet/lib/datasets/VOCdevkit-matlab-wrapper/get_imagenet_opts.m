function synsets = get_imagenet_opts(path)

tmp = pwd;
cd(path);
try
  addpath('data');
  load('meta_det.mat')
catch
  rmpath('data');
  cd(tmp);
  error(sprintf('evaluation directory not found under %s', path));
end
rmpath('data');
cd(tmp);
