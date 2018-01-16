% test retune
addpath('/data3/Zhangqianyu/caffe/matlab/');
name = 'iter_100_20_30';
m = csvread(['./result/' name '/prun_mnist-30.csv']);
% for i = 2:200
for idx = 69:100
    x = m(idx,3:end);
    caffe.reset_all();
    deploy = '../lenetmodel/lenet.prototxt';
    caffemodel = '../lenetmodel/lenet.caffemodel';
    phase = 'train'; % dropout applied
    if ~exist(deploy, 'file')
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
    %calculate size of each layer
    w=cell(1);b = cell(1);layersize = cell(1);
    for i = 1:length(layers)
        w{i}=net.params(layers{i},1).get_data();
        b{i}=net.params(layers{i},2).get_data();
        layersize{i}=size(w{i});
    end
    
    %2. fc pruning;
    for i= 1:length(convlayers)
            ind=find(w{i}<x(i)&w{i}>x(i)*(-1));
            count(i)=length(ind);
            w{i}(ind)=0;
            for j = 1:layersize{i}(4)
                if sum(sum(sum(w{i}(:,:,:,j))))==0
                    b{i}(j)=0;
                end
            end
    end
    
    %save net params
    for i=1:length(layers)
        net.params(layers{i},1).set_data(w{i});
        net.params(layers{i},2).set_data(b{i});
    end
    
    modelname=['child' num2str(idx) '.caffemodel'];
    model_root = '/data3/Zhangqianyu/lab_on_conv/prun_conv/model/pretune/';
    net.save([model_root modelname]);
end
% end