
require 'torch'
require 'dok'

local DataContainer = torch.class('DataContainer')

local function CatNumSize(num,size)
    local stg = torch.LongStorage(size:size()+1)
    stg[1] = num
    for i=2,stg:size() do
        stg[i]=size[i-1]
    end
    return stg
end
function DataContainer:__init(...)
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataContainer ',
    {arg='BatchSize', type='number', help='Number of Elements in each Batch',req = true},
    {arg='PatchSize', type='number',help='Patch size', default = 32},
    {arg='TensorType', type='string', help='Type of Tensor', default = 'torch.FloatTensor'},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='List', type='userdata', help='source of DataContainer', req=true},
    {arg='Data', type='userdata', help='Data', req = true},
    {arg='ListGenFunc', type='function', help='Generate new list'}
    )

    self.BatchSize = args.BatchSize
    self.TensorType = args.TensorType
    self.ExtractFunction = args.ExtractFunction
    self.PatchSize = args.PatchSize
    self.Batch = torch.Tensor():type(self.TensorType)
    self.Data = args.Data
    self.List = args.List
    self.RandPerm = torch.randperm(self.Data:size(1))
    self.ListGenFunc = args.ListGenFunc
    self.NumEachSet = 3--self.List:size(2)
    self:Reset()
end

function DataContainer:Reset()
    self.CurrentItem = 1
end

function DataContainer:size()
    return self.Data:size(1)
end

function DataContainer:Reset()
    self.CurrentItem = 1
end
function DataContainer:ShuffleItems()
    local RandOrder = torch.randperm(self.Data:size(1)):long()
    self.Data= self.Data:indexCopy(1,RandOrder,self.Data)
    self.List = self.List:indexCopy(1,RandOrder,self.List)
    self.RandPerm = torch.randperm(self.Data:size(1))
    print('(DataContainer)===>Shuffling Items')

end


function DataContainer:__tostring__()
    local str = 'DataContainer:\n'
    if self:size() > 0 then
        str = str .. ' + num samples : '.. self:size()
    else
        str = str .. ' + empty set...'
    end
    return str
end

function DataContainer:GenerateList()
    self.List = self.ListGenFunc(self.Data)

end

function DataContainer:GetNextBatch()
    local size = math.min(self:size()-self.CurrentItem + 1, self.BatchSize )
    if size <= 0 then
        return nil
    end

    if self.Batch:dim() == 0 or size < self.BatchSize then
        local nsz = torch.LongStorage({self.NumEachSet, size,3, self.PatchSize,  self.PatchSize})
        self.Batch:resize(nsz)
    end
    local batch_table = {}
    local d = self.Data:narrow(1,self.CurrentItem,size)
    local loc = self.List:narrow(1,self.CurrentItem,size)
    local d_false = self.RandPerm:narrow(1,self.CurrentItem, size)
    for b=1, size do
        local nextb = math.random(self.Data:size(1))
        local patch1 = SamplePatch(d[b], loc[b][1])
        local patch2 = SamplePatch(self.Data[d_false[b]], loc[b][2])
        local patch3 = SamplePatch(d[b], loc[b][3])
        self.Batch:select(1,1):select(1,b):copy(patch1)
        self.Batch:select(1,2):select(1,b):copy(patch2)
        self.Batch:select(1,3):select(1,b):copy(patch3)
    end
    for i=1, self.NumEachSet do
        table.insert(batch_table, self.Batch[i])
    end
    self.CurrentItem = self.CurrentItem + size
    return batch_table
end








