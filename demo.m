function demo()
%     popsize=[150,125,100,75,50,25];
%     niche=[ones(1,6)*20,ones(1,6)*5,floor(ones(1,6).*popsize./2),floor(ones(1,6).*popsize.*3./4)];
%     popsize = repmat(popsize,1,4);
    addpath('/data3/Zhangqianyu/caffe/matlab/');
    deploy = '../lenetmodel/lenet.prototxt';
    caffemodel = '../lenetmodel/lenet.caffemodel';
    % domain = compute_domain(deploy,caffemodel);
	load('domain.mat');
    mop=testmop('nnprun',2,domain(1:2,:));

    % for i=1:24
%     pareto = moead( mop, i,'popsize', popsize(i), 'niche', niche(i), 'iteration', 200, 'method', 'te');
    pareto = moead( mop,'popsize', 100, 'niche', 20, 'iteration', 30, 'method', 'te');%if u change popsize,u should change init_weights.m
% end
end