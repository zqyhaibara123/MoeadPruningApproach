function y = retune(x)
    caffe.reset_all();
    deploy = '../lenetmodel/lenet.prototxt';
    %caffemodel = '../lenetmodel/lenet.caffemodel';
    caffemodel='../lenetmodel/lenet.caffemodel';
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
    
    modelname='childpre.caffemodel';
    model_root = '/data3/Zhangqianyu/lab_on_conv/prun_conv/model/';
    net.save([model_root modelname]);
    
    %retune
    str1=['./build/tools/caffe train '...
        '--solver=/data3/Zhangqianyu/lab_on_conv/prun_conv/retune_solver.prototxt '...
        '--weights=' ];
    str2=' --gpu=0';
    retunecmd = [str1 model_root modelname str2];
    
    cd /data3/Zhangqianyu/caffe_zqy_pruned/
    [~,result] = system(retunecmd,'-echo');%输出到日志,避免因内存不够/造成白retune
    fid = fopen('/data3/Zhangqianyu/lab_on_conv/prun_conv/log.txt','a');
    fprintf(fid,'%s\n',result);
    fclose(fid);
    clear result;
    
    %test
    str1=['./build/tools/caffe test '...
    '--model=/data3/Zhangqianyu/lab_on_conv/lenetmodel/lenet_test.prototxt'];
    str2=' --weights=';
    str3=' --iterations=100 --gpu 0';

    [~,results]=system([str1 str2 model_root 'child_retune_iter_8000.caffemodel' str3]);% epochs=10
    results=results(end-80:end);
    ind=strfind(results,'accuracy');
    %'-echo'
    cd ../lab_on_conv/prun_conv
    accstr1=results(ind(1)+10:end);
    %avoid of wrong, save accuracy str
    y=str2double(accstr1);
 %   accuracy5=str2double(accstr2);
end