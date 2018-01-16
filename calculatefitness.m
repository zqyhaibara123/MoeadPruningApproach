function y = calculatefitness(x)
    %x is a row vector, witch length=num of layers
    % load model
% 	addpath('/disk3/Zhangqianyu/caffe/matlab/');% delete when run demo
    caffe.reset_all();
    model_dir = '../lenetmodel/';
    net_model = [model_dir 'lenet.prototxt'];
    net_weights = [model_dir 'lenet.caffemodel'];
%     net_weights = './model/retune/child1_retune_iter_10000.caffemodel';
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    if ~exist(net_weights, 'file')
      error('Please download CaffeNet from Model Zoo before you run this demo');
    end
    % Initialize a network
    net = caffe.Net(net_model, net_weights, phase);
    %get layers: conv+ip/fc
    layers = net.layer_names;
    convlayers = layers(strncmp('conv',layers,4));
    fclayers = [layers(strncmp('ip',layers,2));layers(strncmp('fc',layers,2))];
    convnum = length(convlayers);
    fcnum = length(fclayers);
    layers = [convlayers;fclayers];
    layernum = convnum;
    %load weights
    %calculate size of each layer
    w=cell(1);b=cell(1);layersize = cell(1);
    for i = 1:length(layers)
        w{i}=net.params(layers{i},1).get_data();
        b{i}=net.params(layers{i},2).get_data();
        layersize{i}=size(w{i});
    end
    featuremapborder=[24,8];
    % mutation prob: average choosen
    
    %2. fc pruning;
    for i= 1:convnum
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
    for i=1:layernum
        net.params(layers{i},1).set_data(w{i});
        net.params(layers{i},2).set_data(b{i});
    end
    newmodel='./model/child.caffemodel';
    model_root = '/data3/Zhangqianyu/lab_on_conv/prun_conv/model/child.caffemodel';
    net.save(newmodel);
    
    accuracy=trainalex(model_root);% 0.9912
    %calculate y1, y2, y3
    y=zeros(2,1);
    y(1) = 1-accuracy;
    %time cost
    % ground truth:  1.448813632000000e+09
    for i = 1:convnum
        y(2) = y(2) + featuremapborder(i)*featuremapborder(i)*2*(layersize{i}(1)*layersize{i}(2)*layersize{i}(3)*layersize{i}(4)-count(i));
    end
    y(2)=fix(y(2)/37760)/100;
end

function [accuracy1]=trainalex(model_root)
    str1=['./build/tools/caffe test '...
    '--model=/data3/Zhangqianyu/lab_on_conv/lenetmodel/lenet_test.prototxt --weights='];
    str2=' --iterations=100 --gpu 0';
    
    cd /data3/Zhangqianyu/caffe/
    [~,results]=system([str1 model_root str2]);
    results=results(end-80:end);
    ind=strfind(results,'accuracy');
    %'-echo'
    cd ../lab_on_conv/prun_conv/
    accstr1=results(ind(1)+10:end);
    %avoid of wrong, save accuracy str
    accuracy1=str2double(accstr1);
 %   accuracy5=str2double(accstr2);
end
