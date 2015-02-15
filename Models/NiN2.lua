
require 'cunn'
require 'ccn2'
local model = nn.Sequential() 

model:add(nn.Transpose({1,4},{1,3},{1,2}))

model:add(ccn2.SpatialConvolution(3, 192, 5, 1, 2,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 160, 1, 1,0,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(160, 96, 1, 1,0,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(3, 2))
model:add(nn.Dropout(0.5))

model:add(ccn2.SpatialConvolution(96, 192, 5, 1, 2,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1,0,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1,0,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(3, 2))
model:add(nn.Dropout(0.5))

model:add(ccn2.SpatialConvolution(192, 192, 3, 1, 1,1,4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1,0,1,4))
model:add(nn.ReLU())
model:add(nn.Transpose({4,1},{4,2},{4,3}))

model:add(nn.SpatialConvolutionMM(192,10, 1,1 ))
model:add(nn.ReLU())

model:add(nn.SpatialAveragePooling(8,8))
model:add(nn.View(10))
model:add(nn.LogSoftMax())

for i,layer in ipairs(model.modules) do
   if layer.bias then
      layer.bias:fill(0)
      layer.weight:normal(0, 0.05)
   end
end


return model
