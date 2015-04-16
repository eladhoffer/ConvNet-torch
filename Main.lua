require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'Model.lua',            'Model file - must return valid network.')
cmd:option('-LR',                 0.1,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                  'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-simplenorm',         false,                  '1 - normalize using only 1 mean and std values, 2 for global contrast normalization')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          1,                      'visualizing results')

opt = cmd:parse(arg or {})
opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.augment then
    require 'image'
end
----------------------------------------------------------------------
-- Model + Loss:
local model = require(opt.network)
local loss = nn.ClassNLLCriterion()
-- classes
local data = require 'Data'
local classes      = data.Classes

local ccn2_compatibility = false
--model:for_each(function(x) 
--    ccn2_compatibility = ccn2_compatibility or (torch.type(x):find('ccn2') ~= nil)
--end)

----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/ConfigLog.log', opt)
local weights_filename = paths.concat(opt.save, 'Weights.t7')
local log_filename = paths.concat(opt.save,'TrainingLog.log')
local Log = optim.Logger(log_filename)
----------------------------------------------------------------------
print '==> Network'
print(model)
print '==> Loss'
print(loss)

----------------------------------------------------------------------

local TensorType = 'torch.FloatTensor'
if opt.type =='cuda' then
    model:cuda()
    loss = loss:cuda()
    TensorType = 'torch.CudaTensor'
end

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

local function updateConfusion(y,yt)
    confusion:batchAdd(y,yt)
end

---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    ccn2_compatibility = true
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('==>Loaded Weights from: ' .. opt.load)
    Weights:copy(w)
end

if opt.nGPU > 1 then
    model:syncParameters()
end


local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
}

local function SampleImages(images,labels)
    if not opt.augment then
        return images,labels
    else

        local sampled_imgs = images:clone()
        for i=1,images:size(1) do
            local sz = math.random(9) - 1
            local hflip = math.random(2)==1

            local startx = math.random(sz) 
            local starty = math.random(sz) 
            local img = images[i]:narrow(2,starty,32-sz):narrow(3,startx,32-sz)
            if hflip then
                img = image.hflip(img)
            end
            img = image.scale(img,32,32)
            sampled_imgs[i]:copy(img)
        end
        return sampled_imgs,labels
    end
end
    

------------------------------
local function Train(Data)

    model:training()

    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = Data,
        ExtractFunction = SampleImages,
        TensorType = TensorType
    }

    local yt = MiniBatch.Labels
    local x = MiniBatch.Data
    local SizeData = Data:size()
    local NumSamples = 0
    local NumBatches = 0
    while MiniBatch:GetNextBatch() do
        NumSamples = NumSamples+x:size(1)
        NumBatches = NumBatches + 1
        if ccn2_compatibility==false or math.fmod(x:size(1),32)==0 then
            
            if opt.nGPU > 1 then
               model:zeroGradParameters()
               model:syncParameters()
           end

           local y = optimizer:optimize(x, yt)
           
           updateConfusion(y,yt)
        end
        xlua.progress(NumSamples, SizeData)

        if math.fmod(NumBatches,100)==0 then
            collectgarbage()
        end
    end

end
------------------------------
local function Test(Data)

    model:evaluate()

    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = Data,
        TensorType = TensorType
    }

    local yt = MiniBatch.Labels
    local x = MiniBatch.Data
    local SizeData = Data:size()
    local NumSamples = 0
    local NumBatches = 0
    while MiniBatch:GetNextBatch() do
        NumSamples = NumSamples+x:size(1)
        NumBatches = NumBatches + 1
        if ccn2_compatibility==false or math.fmod(x:size(1),32)==0 then
            local y = model:forward(x)
            updateConfusion(y,yt)
        end
        xlua.progress(NumSamples, SizeData)

        if math.fmod(NumBatches,500)==0 then
           -- image.display(x)
            collectgarbage()
        end
    end

end

local epoch = 1
print '\n==> Starting Training\n'
while true do
    data.TrainData:ShuffleItems()

    print('Epoch ' .. epoch) 
    confusion:zero()
    Train(data.TrainData)
    torch.save(weights_filename, Weights)
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end
    confusion:zero()
    Test(data.TestData)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end
    print('Training Error = ' .. ErrTrain)
    print('Test Error = ' .. ErrTest)
    Log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
        Log:plot()
    end
    epoch = epoch+1
end
