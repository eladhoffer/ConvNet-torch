require 'eladtools'
local opt = opt or {}
local Dataset = opt.dataset or 'Cifar10'
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = opt.datapath or '/home/ehoffer/Datasets/'
local SimpleNormalization = (opt.normalize==1) or false

local TestData
local TrainData
local Classes

if Dataset =='Cifar100' then
    TrainData = torch.load(DataPath .. 'Cifar100/cifar100-train.t7')
    TestData = torch.load(DataPath .. 'Cifar100/cifar100-test.t7')
    TrainData.labelCoarse:add(1)
    TestData.labelCoarse:add(1)
    Classes = torch.linspace(1,100,100):storage():totable()
elseif Dataset == 'Cifar10' then
    TrainData = torch.load(DataPath .. 'Cifar10/cifar10-train.t7')
    TestData = torch.load(DataPath .. 'Cifar10/cifar10-test.t7')
    Classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
elseif Dataset == 'STL10' then
    TrainData = torch.load(DataPath .. 'STL10/stl10-train.t7')
    TestData = torch.load(DataPath .. 'STL10/stl10-test.t7')
    Classes = {'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'}
    TestData.label = TestData.label:add(-1):byte()
    TrainData.label = TrainData.label:add(-1):byte()
elseif Dataset == 'MNIST' then
    TrainData = torch.load(DataPath .. 'MNIST/mnist-train.t7')
    TestData = torch.load(DataPath .. 'MNIST/mnist-test.t7')
    Classes = {1,2,3,4,5,6,7,8,9,0}
    TestData.data = TestData.data:view(TestData.data:size(1),1,28,28)
    TrainData.data = TrainData.data:view(TrainData.data:size(1),1,28,28)
    TestData.label = TestData.label:byte()
    TrainData.label = TrainData.label:byte()
elseif Dataset == 'SVHN' then
    TrainData = torch.load(DataPath .. 'SVHN/train_32x32.t7','ascii')
    ExtraData = torch.load(DataPath .. 'SVHN/extra_32x32.t7','ascii')
    TrainData.X = torch.cat(TrainData.X, ExtraData.X,1)
    TrainData.y = torch.cat(TrainData.y[1], ExtraData.y[1],1)
    TrainData = {data = TrainData.X, label = TrainData.y}
    TrainData.label = TrainData.label:add(-1):byte()
    TrainData.X = nil
    TrainData.y = nil
    ExtraData = nil

    TestData = torch.load(DataPath .. 'SVHN/test_32x32.t7','ascii')
    TestData = {data = TestData.X, label = TestData.y[1]}
    TestData.label = TestData.label:add(-1):byte()
    Classes = {1,2,3,4,5,6,7,8,9,0}
end
TrainData.label:add(1)
TestData.label:add(1)



TrainData.data = TrainData.data:float()
TestData.data = TestData.data:float()
local _, channels, y_size, x_size = unpack(TrainData.data:size():totable())
if SimpleNormalization then
    local mean = TrainData.data:mean()
    local std = TrainData.data:std()

    TrainData.data:add(-mean):div(std)
    TestData.data:add(-mean):div(std)
else
    --Preprocesss
    local meansfile = paths.concat(PreProcDir,'means.t7')
    if Whiten then
        require 'unsup'
        local means, P, invP
        local Pfile = paths.concat(PreProcDir,'P.t7')
        local invPfile = paths.concat(PreProcDir,'invP.t7')

        if (paths.filep(Pfile) and paths.filep(invPfile) and paths.filep(meansfile)) then
            P = torch.load(Pfile)
            invP = torch.load(invPfile)
            means = torch.load(meansfile)
            TrainData.data = unsup.zca_whiten(TrainData.data, means, P, invP)
        else
            TrainData.data, means, P, invP = unsup.zca_whiten(TrainData.data)
            torch.save(Pfile,P)
            torch.save(invPfile,invP)
            torch.save(meansfile,means)
        end
        TestData.data = unsup.zca_whiten(TestData.data, means, P, invP)


        TrainData.data = TrainData.data:float()
        TestData.data = TestData.data:float()

    else
        local means, std
        local loaded = false
        local stdfile = paths.concat(PreProcDir,'std.t7')
        if paths.filep(meansfile) and paths.filep(stdfile) then
            means = torch.load(meansfile)
            std = torch.load(stdfile)
            loaded = true
        end
        if not loaded then
            means = torch.mean(TrainData.data, 1):squeeze()
        end
        TrainData.data:add(-1, means:view(1,channels,y_size,x_size):expand(TrainData.data:size(1),channels,y_size,x_size))
        TestData.data:add(-1, means:view(1,channels,y_size,x_size):expand(TestData.data:size(1),channels,y_size,x_size))

        if not loaded then
            std = torch.std(TrainData.data, 1):squeeze()
        end
        TrainData.data:cdiv(std:view(1,channels,y_size,x_size):expand(TrainData.data:size(1),channels,y_size,x_size))
        TestData.data:cdiv(std:view(1,channels,y_size,x_size):expand(TestData.data:size(1),channels,y_size,x_size))

        if not loaded then
            torch.save(meansfile,means)
            torch.save(stdfile,std)
        end

    end
end

local TrainDataProvider = DataProvider{
    Name = 'TrainingData',
    CachePrefix = nil,
    CacheFiles = false,
    Source = {TrainData.data,TrainData.label},
    MaxNumItems = 1e6,
    CopyData = false,
    TensorType = 'torch.FloatTensor',

}
local TestDataProvider = DataProvider{
    Name = 'TestData',
    CachePrefix = nil,
    CacheFiles = false,
    Source = {TestData.data, TestData.label},
    MaxNumItems = 1e6,
    CopyData = false,
    TensorType = 'torch.FloatTensor',

}

return{
    TrainData = TrainDataProvider,
    TestData = TestDataProvider,
    Classes = Classes
}
