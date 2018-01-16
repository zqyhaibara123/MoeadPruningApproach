function domain = compute_domain(deploy,caffemodel)
    %每层至少有一个连接
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    if ~exist(caffemodel, 'file')
      error('Please download CaffeNet from Model Zoo before you run this demo');
    end
    % Initialize a network
    net = caffe.Net(deploy, caffemodel, phase);
    %get layers: conv+ip/fc
    layers = net.layer_names;
    convlayers = layers(strncmp('conv',layers,4));
    fclayers = [layers(strncmp('ip',layers,2));layers(strncmp('fc',layers,2))];
    layers = [convlayers;fclayers];
    %load weights
    w=cell(1);
    for i = 1:length(layers)
        w{i}=net.params(layers{i},1).get_data();
    end
    domain = [];
    for i = 1:length(convlayers)
        lb = min(min(min(min(w{i}))));
        ub = max(max(max(max(w{i}))));
        %ub = max(abs(lb),ub);
        %lb=0;
        domain = [domain;lb ub];
    end
    for i =1+length(convlayers):length(fclayers)+length(convlayers)
        lb = min(min(w{i}));
        ub = max(max(w{i}));
        %ub = max(abs(lb),max(max(w{i})));
        %lb = 0;
        domain = [domain;lb ub];
    end
end